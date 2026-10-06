export const researchArticles = [
  { slug: "pretraining-hobbylm", number: "01", title: "Pretraining HobbyLM from scratch", summary: "The data, infrastructure, training decisions and lessons behind building HobbyLM’s foundation.", status: "Published" },
  { slug: "hobbylm-architecture", number: "02", title: "HobbyLM architecture", summary: "A technical account of the model architecture and the decisions behind it.", status: "In preparation" },
  { slug: "post-training-and-sft", number: "03", title: "Post-training and SFT", summary: "The instruction-tuning process, its constraints, and what changed after pretraining.", status: "In preparation" },
  { slug: "evaluation-methodology", number: "04", title: "Evaluation methodology and results", summary: "Protocols, approved measurements, and known limitations presented with their context.", status: "Evidence pending" },
] as const;

export const founders = [
  { id: "prajhan", name: "Prabhu Rajhan @ Prajhan", alt: "Engraved black-and-white portrait of Prabhu Rajhan" },
  { id: "hariharan", name: "Hariharan", alt: "Engraved black-and-white portrait of Hariharan" },
] as const;
