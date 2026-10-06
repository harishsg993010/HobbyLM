import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { SignalHeading } from "@/components/typography/SignalHeading";
import { ArticleContents } from "@/components/research/ArticleContents";
import { ArticleFigure } from "@/components/research/ArticleFigure";
import { pretrainingFigures } from "@/content/pretrainingFigures";
import { researchArticles } from "@/content/site";
import styles from "./page.module.css";

export function generateStaticParams() { return researchArticles.filter((article) => article.status === "Published").map(({ slug }) => ({ slug })); }

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const article = researchArticles.find((item) => item.slug === slug);
  return article?.status === "Published" && slug === "pretraining-hobbylm" ? { title: article.title, alternates: { canonical: `/research/${article.slug}` } } : {};
}

export default async function ArticlePage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const article = researchArticles.find((item) => item.slug === slug);
  if (!article) notFound();
  if (article.status !== "Published" || slug !== "pretraining-hobbylm") notFound();

  const sections = [
    { id: "derisking", label: "Derisking the full run" },
    { id: "data-and-model", label: "Data and model design" },
    { id: "training", label: "Main pretraining and annealing" },
    { id: "checkpoint", label: "Selecting the release checkpoint" },
    { id: "release", label: "What we are releasing" },
    { id: "limitations", label: "Limitations and lessons" },
  ] as const;
  const figures = Object.fromEntries(pretrainingFigures.map((figure) => [figure.id, figure]));

  return <article className={`section ${styles.article}`}>
    <header className={`container ${styles.header}`}>
      <SignalHeading as="h1" className="display">{article.title}</SignalHeading>
      <p className={`intro-copy ${styles.deck}`}>The data, infrastructure and engineering decisions behind building HobbyLM&apos;s foundation.</p>
    </header>
    <div className={`container ${styles.articleGrid}`}>
      <ArticleContents sections={sections} />
      <div className={`article-body-copy ${styles.body}`}>
        <p className={styles.lead}>We pretrained HobbyLM, a sparse mixture-of-experts base model, on approximately 100 billion tokens. The main phase ran for about 66 hours on four H200 GPUs, followed by 10 hours of annealing. A smaller proxy run exposed a problem in how data reached the model before we launched the full run. Later, FineWeb validation loss and downstream evaluation disagreed about which checkpoint to release.</p>
        <p>This article covers the decisions behind that run: how we used proxy experiments, tested the training system, structured the model and selected the final checkpoint. It also defines the release boundary. We are publishing the architecture, model and selected checkpoints. The exact data and optimization recipe remains internal.</p>
        <p>In this article, &ldquo;from scratch&rdquo; refers to the model weights. Pretraining began from initialization rather than from an existing language-model checkpoint. We still used established public datasets, a GPT-2 tokenizer configuration and standard transformer components. Our work focused on assembling those parts into a sparse MoE model and a training path that could run, recover, and leave behind inspectable evidence.</p>
        <ArticleFigure figure={figures["pretraining-run-summary"]} />

        <section id="derisking">
          <h2>Derisking the full run</h2>
          <p>Before committing four H200s to the main run, we tested the pipeline on a smaller proxy model. Its training loss fell quickly, but broader evaluation moved in the wrong direction.</p>
          <p>The source datasets were individually valid. The problem was their delivery. The model saw extended stretches from one source family before moving to another, so the batches did not represent the distribution we intended to train on over time. Falling loss showed that the model was learning the current batches. It said little about whether behavior was holding up across the wider corpus.</p>
          <ArticleFigure figure={figures["pretraining-data-delivery"]} />
          <p>That result clarified what we needed from a proxy run. A small model could not forecast the final model&apos;s benchmark scores with much precision. It could test the machinery around the model while failures were still cheap: data loading, ordering, validation, checkpointing and recovery.</p>
          <p>We corrected the delivery behavior and repeated the proxy experiment. General evaluation recovered enough to proceed with the larger pipeline. The implementation is internal. The public lesson is to inspect the sequence the model receives alongside the list of datasets in its configuration.</p>
          <p>The proxy result also changed how we read training loss. We continued to use it as an optimization signal, but no longer treated a falling curve as sufficient evidence that the complete training system was improving.</p>
          <h3>Infrastructure gates</h3>
          <p>The first infrastructure tests failed before producing a training loss. GPU processes stayed alive, making the job appear active, but the run had not reached a useful step.</p>
          <p>We isolated the launch path and retested it in stages. A short single-GPU run had to produce a loss and write a checkpoint. A multi-GPU run then had to demonstrate that the distributed configuration worked. The final sanity run used the complete configuration and real staged data.</p>
          <p>Each test had explicit exit conditions. The process needed to train, validate, save a checkpoint and show the expected GPU activity. The preflight also checked available storage, experiment-tracking authentication and remote checkpoint backup. We tested recovery before starting the main job instead of assuming that a saved file would work after a failure.</p>
          <ArticleFigure figure={figures["pretraining-infrastructure-gates"]} />
          <p>These checks took time, but each was cheaper than the full run it protected. By launch, the code, data loader, validation path and checkpoint flow had already worked together under realistic conditions.</p>
        </section>

        <section id="data-and-model">
          <h2>Data and model design</h2>
          <h3>Data pool</h3>
          <p>The main corpus drew from four source families: FineWeb-Edu, DCLM, code and FineMath. Together they supplied educational web text, broader web text, program code and mathematical material. Annealing used the narrower Cosmopedia-v2 and FineMath-4plus source families.</p>
          <p>All text passed through the GPT-2 byte-level BPE tokenizer configuration used by the released model. Before training, we checked that the loader could read the prepared data, that the expected sources were present and that samples decoded correctly. Validation data remained separate from training data.</p>
          <p>We relied heavily on filtering and deduplication performed by the upstream dataset maintainers. We did not run one uniform project-level deduplication pass across every source. The phrase &ldquo;approximately 100 billion tokens&rdquo; therefore describes the amount consumed by the training schedule. It does not mean 100 billion unique or uniformly cleaned tokens.</p>
          <h3>Architecture</h3>
          <p>HobbyLM is a decoder-only sparse MoE transformer with 20 layers. Layer 0 has a dense feed-forward block. Layers 1 through 19 use MoE blocks.</p>
          <p>Each MoE block has 64 routed experts and one shared expert. For every token, the router selects eight routed experts while the shared expert remains active. This gives the model access to a larger parameter pool than it evaluates in a single forward pass.</p>
          <p>The attention blocks use grouped-query attention with 16 query heads and eight key-value heads. The public configuration also includes RMSNorm, per-head QK normalization, SwiGLU feed-forward blocks, RoPE with a base of 10,000, and tied token-embedding and language-model-head weights. The tokenizer configuration has a padded vocabulary of 50,304 tokens, and the model was pretrained with a 1,024-token context window.</p>
          <p>The router uses sigmoid scores and an auxiliary-loss-free expert-bias mechanism. The bias affects expert selection but is stored as a buffer rather than a trainable parameter.<sup><a href="#note-1" aria-label="Footnote 1">1</a></sup></p>
          <ArticleFigure figure={figures["pretraining-architecture"]} />
        </section>

        <section id="training">
          <h2>Main pretraining and annealing</h2>
          <p>The two phases consumed approximately 100 billion tokens. The main phase ran for about 66 hours on four H200 GPUs, or approximately 264 GPU-hours.</p>
          <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="Training phase comparison"><table><thead><tr><th>Phase</th><th>Hardware</th><th>Approx. duration</th><th>Approx. GPU-hours</th><th>Final FineWeb validation loss</th></tr></thead><tbody><tr><td>Main pretraining</td><td>4× H200</td><td>66 hours</td><td>264</td><td>3.4112</td></tr><tr><td>Annealing</td><td>4× H200</td><td>10 hours</td><td>40</td><td>3.5487</td></tr><tr><td>Complete schedule</td><td>4× H200</td><td>76 hours</td><td>304</td><td>3.5487</td></tr></tbody></table></div>
          <p>The duration figures cover the two training phases end to end and are intentionally approximate. GPU-hours are the number of GPUs multiplied by each phase&apos;s elapsed duration.</p>
          <p>The original main-run log contains 8,107 sampled training records, 324 periodic validation measurements and 82 checkpoint-save records. It covers the run from initialization through the final main-phase checkpoint. Because the raw log survived, the main validation trajectory can be reconstructed without relying on live dashboard access.</p>
          <ArticleFigure figure={figures["pretraining-validation"]} />
          <p>FineWeb validation loss was 4.3557 at the first recorded validation point and generally declined over the run. Individual measurements rose and fell as the validation sample and nearby training batches changed in difficulty. The final main-phase measurement was 3.4112. The broad trend supports continued optimization; an isolated rise in the curve is not evidence of instability by itself.</p>
          <p>Checkpointing continued throughout the phase. Important milestones and the final state were backed up remotely, which gave us recovery points as well as candidates for later evaluation.</p>
          <h3>Annealing</h3>
          <p>Annealing resumed from the main-phase checkpoint and moved to the narrower Cosmopedia-v2 and FineMath-4plus source families. We verified that the process resumed from the trained state rather than initialization. The phase added about 10 hours, bringing the complete schedule to roughly 76 hours and 304 GPU-hours.</p>
          <p>The change in data distribution mattered when we evaluated the result. The immutable final <code>result.json</code> records a FineWeb validation loss of 3.5487, higher than the main-phase endpoint of 3.4112. FineWeb loss alone would therefore have favored the earlier checkpoint.</p>
          <p>The downstream comparison pointed the other way.</p>
        </section>

        <section id="checkpoint">
          <h2>Selecting the release checkpoint</h2>
          <p>We evaluated the pre-annealing and post-annealing checkpoints on the same seven-task zero-shot suite: HellaSwag, OpenBookQA, WinoGrande, ARC Challenge, ARC Easy, BoolQ and PIQA. For each task, the reported score uses normalized accuracy where available and accuracy otherwise. The headline number is the unweighted mean of the seven task metrics.</p>
          <p>The surviving records report an average of 44.71 before annealing and 47.61 afterward, a gain of 2.90 points. Six of the seven task scores increased. BoolQ fell from 56.24 to 49.54.</p>
          <div className={styles.tableWrap} tabIndex={0} role="region" aria-label="Checkpoint evaluation comparison"><table><thead><tr><th>Reported measure</th><th>Before annealing</th><th>After annealing</th><th>Change</th></tr></thead><tbody><tr><td>Seven-task average</td><td>44.71</td><td>47.61</td><td>+2.90</td></tr><tr><td>BoolQ</td><td>56.24</td><td>49.54</td><td>−6.70</td></tr><tr><td>FineWeb validation loss</td><td>3.4112</td><td>3.5487</td><td>+0.1375</td></tr></tbody></table></div>
          <ArticleFigure figure={figures["pretraining-checkpoint-selection"]} />
          <p>Lower is better for validation loss; higher is better for the reported task scores. The table places the measurements together because their disagreement determined the release choice, not because their numerical scales are directly comparable.</p>
          <p>We chose the annealed checkpoint because it had the higher average under this protocol. This is a specific release decision, not a claim that the checkpoint is better on every task or distribution. The suite contains seven zero-shot tasks, and one of them declined.</p>
          <p>FineWeb validation loss and the downstream suite measure different things. The first measures next-token prediction on a held-out web distribution. The second samples performance across seven task formats. Annealing changed the training distribution toward synthetic textbook and mathematical material, so the divergence is plausible. We did not run the controlled ablation required to attribute the BoolQ decline, or the aggregate increase, to a particular part of that mixture.</p>
          <p>There is also an evidence limitation. We retained the task list, zero-shot setting and aggregation procedure in code, while the scores survived in project records and the released model card. The original machine-readable evaluation JSON and exact package version were not retained. We therefore treat these as reported evaluation results, not a reproduced evaluation.</p>
          <p>The two checkpoints are still useful together. Their disagreement shows why checkpoint selection needs a declared objective. A held-out loss can track optimization on its own distribution without determining which model performs better under a separate downstream protocol.</p>
        </section>

        <section id="release">
          <h2>What we are releasing</h2>
          <p>The public release includes:</p>
          <ul>
            <li>The approved architecture and implementation.</li>
            <li>The <a href="https://huggingface.co/harims95/hobbylm-1b-hf">HobbyLM model</a> and GPT-2 tokenizer configuration.</li>
            <li><a href="https://huggingface.co/harims95/hobbylm-1b-checkpoints">Selected milestone checkpoints</a> from pretraining and annealing.</li>
            <li>The final <code>result.json</code> and the main training log.</li>
            <li>The approved <a href="https://huggingface.co/datasets/harims95/hobbylm-routing-dynamics">routing-dynamics dataset</a>.</li>
          </ul>
          <p>The release contains selected checkpoints rather than every state saved during training. A single important checkpoint is approximately 8.5 GB, and an exhaustive upload would add substantial storage while revealing more of the internal training trajectory. The selected states cover meaningful stages for inspection and evaluation.</p>
          <p>We name the approved source families but do not redistribute upstream text. Exact data mixture and training-recipe details (including sampling, optimizer and learning-rate construction, and efficiency techniques) remain internal. This boundary reflects both upstream licences and our own development roadmap, and will apply similarly to future work on context extension, supervised fine-tuning and reinforcement fine-tuning.</p>
          <p>The open artifacts let researchers inspect the architecture, run the model and compare selected stages, but they do not reproduce Fuel Labs&apos; complete data and optimization pipeline.</p>
        </section>

        <section id="limitations">
          <h2>Limitations and lessons</h2>
          <p>The data total is a measure of scheduled consumption, not a count of unique documents. Cleaning and deduplication varied by upstream source because the project did not apply one uniform pass to the entire pool. The exact mixture, scheduling and preparation process is not public.</p>
          <p>The checkpoint comparison is also narrow. It covers seven zero-shot tasks and has not been independently reproduced. The original evaluation JSON and exact package version are missing, so the published values remain reported results. The annealing time series was not retained; only its final validation result is presented here.</p>
          <p>Three practices carried most of the engineering value from this run.</p>
          <p>First, a proxy model is most useful when it tests the whole path around training. The smaller run found a data-delivery problem at a scale where we could still correct and repeat the experiment.</p>
          <p>Second, expensive jobs need observable gates. A live process is not proof of progress. Our launch checks required a loss, validation, checkpoint output and expected hardware activity before the full configuration passed.</p>
          <p>Third, evaluation output deserves the same retention discipline as model weights. Future runs should save the command, model revision, framework revision, raw JSON and a short manifest showing how each headline number was calculated. Those files are tiny beside a checkpoint and would have made the pre/post comparison independently reconstructable.</p>
          <p>The main result is a trained model, but the more reusable outcome is the procedure that allowed us to trust the run. We tested the data stream before scaling it, required the system to prove that it could train and recover, and selected the release checkpoint against an explicit evaluation objective. Where the record does not support a broader conclusion, we have left it open.</p>
        </section>

        <footer className={styles.notes} aria-label="Article notes">
          <p id="note-1"><sup>1</sup> One item of Hub metadata counts 1,216 more serialized elements than the verified trainable-parameter total. The difference is exactly 19 MoE layers multiplied by 64 expert-bias values. The native and Hugging Face trainable-parameter totals otherwise match.</p>
        </footer>
      </div>
    </div>
  </article>;
}
