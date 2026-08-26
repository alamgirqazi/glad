<h1 align="center" style="font-weight: bold; font-size: 2.5rem;"> Adversarial Robustness of Retrieval-Based Fact-Checking: <span style="font-size: 1.5rem; font-weight: normal; color: #555;">Detecting and Mitigating Evidence Contamination</span> </h1> <div align="center">
Alamgir Munir Qazi<sup>1</sup>   John P. McCrae<sup>2</sup>   Jamal Abdul Nasir<sup>1</sup>

<sup>1</sup>School of Computer Science, University of Galway, Ireland <sup>2</sup>Research Ireland Insight Centre and ADAPT Centre, University of Galway, Ireland

</div> <div class="content has-text-justified"> <p> Retrieval-based fact-checking systems ground verification in large document collections, creating a critical vulnerability: adversarial evidence contamination. We demonstrate that factually inconsistent but contextually coherent evidence generated to satisfy both retrieval-similarity and linguistic-fluency constraints can systematically mislead verification models. Contaminating just 3.1-7.4% of training corpora with five label-specific strategies causes adversarial content to penetrate up to 35.2% of top-k retrievals, degrading verification accuracy by up to 46.9% relative F1. Existing retrieval-stage defenses, validated against gradient-optimized or syntactically anomalous attacks, transfer poorly to this threat model: detection rates collapse and F1 recovery exhibits high variance across seeds. We propose GLAD (Geometric-Lexical Adversarial Defense), which fuses dense embedding peer-coherence with lexical distributional isolation. Across two benchmarks, two embedding models, and five random seeds, GLAD achieves the highest detection rates (70-86%) and the most consistent F1 recovery, reclaiming 73-80% of contamination-induced degradation on LIAR-RAW while approaching clean baselines on RAWFC. Crucially, unlike defenses that trade clean performance for robustness, GLAD imposes negligible cost on uncontaminated corpora. </p> </div>


## Code

coming soon.

## Dataset

coming soon.
