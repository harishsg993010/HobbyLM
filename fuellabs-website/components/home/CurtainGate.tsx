"use client";

import dynamic from "next/dynamic";

const CurtainReveal = dynamic(() => import("./CurtainReveal").then((module) => module.CurtainReveal), { ssr: false });

export function CurtainGate() { return <CurtainReveal />; }
