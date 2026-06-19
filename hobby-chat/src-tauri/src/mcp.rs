//! MCP (Model Context Protocol) client via the official `rmcp` SDK. Spawns local stdio MCP servers
//! (e.g. `npx -y @modelcontextprotocol/server-filesystem …`), lists their tools, and routes
//! tools/call. The chat tool-loop exposes these alongside the built-ins. Config mirrors Claude
//! Desktop's `mcpServers`.
//!
//! rmcp is async; we own a dedicated multi-thread tokio runtime and `block_on` it. block_on must NOT
//! run on a tokio worker thread, so callers invoke these from a plain std::thread (the chat worker, or
//! a spawned thread in the commands).

use anyhow::{anyhow, Result};
use rmcp::{
    model::CallToolRequestParam,
    service::{RoleClient, RunningService, ServiceExt},
    transport::{ConfigureCommandExt, TokioChildProcess},
};
use serde_json::{json, Value};
use tokio::process::Command as TokioCommand;

#[derive(Clone, serde::Serialize)]
pub struct McpTool {
    pub server: String,
    pub name: String,
    pub description: String,
    pub schema: Value, // JSON-Schema inputSchema
    pub enabled: bool, // only enabled tools are exposed to the model (500M can't handle many)
}

struct Server {
    name: String,
    svc: RunningService<RoleClient, ()>,
    tools: Vec<McpTool>,
}

pub struct McpManager {
    rt: tokio::runtime::Runtime,
    servers: Vec<Server>,
}

impl McpManager {
    pub fn new() -> Result<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        Ok(McpManager { rt, servers: Vec::new() })
    }

    /// Connect (or reconnect) a stdio MCP server and cache its tools. Returns the tool count.
    pub fn add(&mut self, name: &str, command: &str, args: &[String]) -> Result<usize> {
        self.servers.retain(|s| s.name != name);

        // On Windows, npx/uvx/node are .cmd shims — launch via `cmd /C` so PATH resolves them.
        let (program, full_args): (String, Vec<String>) = if cfg!(windows) {
            let mut v = vec!["/C".to_string(), command.to_string()];
            v.extend(args.iter().cloned());
            ("cmd".to_string(), v)
        } else {
            (command.to_string(), args.to_vec())
        };
        let name_owned = name.to_string();

        let (svc, tools) = self.rt.block_on(async move {
            let transport = TokioChildProcess::new(
                TokioCommand::new(&program).configure(|c| {
                    c.args(&full_args);
                }),
            )?;
            let svc = ().serve(transport).await?;
            let listed = svc.list_tools(Default::default()).await?;
            // Read fields via JSON to stay robust across rmcp model-type changes.
            let lv = serde_json::to_value(&listed).unwrap_or_else(|_| json!({}));
            let mut tools: Vec<McpTool> = lv
                .get("tools")
                .and_then(|t| t.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|t| McpTool {
                            server: name_owned.clone(),
                            name: t.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                            description: t.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                            schema: t.get("inputSchema").cloned().unwrap_or_else(|| json!({})),
                            enabled: false,
                        })
                        .collect()
                })
                .unwrap_or_default();
            // Default ON only for small servers — the 500M model degrades past ~6 tools, so large
            // servers start fully disabled and the user enables the handful they want.
            let default_on = tools.len() <= 6;
            for t in tools.iter_mut() {
                t.enabled = default_on;
            }
            anyhow::Ok((svc, tools))
        })?;

        let n = tools.len();
        self.servers.push(Server { name: name.to_string(), svc, tools });
        Ok(n)
    }

    pub fn remove(&mut self, name: &str) {
        self.servers.retain(|s| s.name != name);
    }

    pub fn all_tools(&self) -> Vec<McpTool> {
        self.servers.iter().flat_map(|s| s.tools.iter().cloned()).collect()
    }

    /// Only the tools the user has enabled — this is what's exposed to the model.
    pub fn enabled_tools(&self) -> Vec<McpTool> {
        self.servers.iter().flat_map(|s| s.tools.iter().filter(|t| t.enabled).cloned()).collect()
    }

    pub fn set_tool_enabled(&mut self, server: &str, tool: &str, enabled: bool) {
        for s in self.servers.iter_mut() {
            if s.name == server {
                for t in s.tools.iter_mut() {
                    if t.name == tool {
                        t.enabled = enabled;
                    }
                }
            }
        }
    }

    pub fn server_summaries(&self) -> Vec<Value> {
        self.servers
            .iter()
            .map(|s| {
                json!({
                    "server": s.name,
                    "tools": s.tools.iter().map(|t| json!({ "name": t.name, "enabled": t.enabled })).collect::<Vec<_>>(),
                })
            })
            .collect()
    }

    /// Call a tool by name on whichever server provides it (tolerant of model whitespace/casing).
    /// Returns the joined text content.
    pub fn call(&self, tool: &str, args: &Value) -> Result<String> {
        let want = tool.trim().to_lowercase();
        let (server, actual) = self
            .servers
            .iter()
            .find_map(|s| {
                s.tools
                    .iter()
                    .find(|t| t.name.to_lowercase() == want)
                    .map(|t| (s, t.name.clone()))
            })
            .ok_or_else(|| anyhow!("no connected MCP server provides tool `{tool}`"))?;
        let name = actual; // use the server's exact tool name
        let arguments = args.as_object().cloned();

        let res = self.rt.block_on(async {
            let mut param = CallToolRequestParam::new(name);
            if let Some(a) = arguments {
                param = param.with_arguments(a);
            }
            server.svc.call_tool(param).await
        })?;

        // Extract text content via JSON (content: [{type:"text", text:"…"}]).
        let rv = serde_json::to_value(&res).unwrap_or_else(|_| json!({}));
        let mut out = String::new();
        if let Some(arr) = rv.get("content").and_then(|c| c.as_array()) {
            for item in arr {
                if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                    out.push_str(t);
                    out.push('\n');
                }
            }
        }
        if out.trim().is_empty() {
            out = rv.to_string();
        }
        Ok(out.trim().to_string())
    }
}
