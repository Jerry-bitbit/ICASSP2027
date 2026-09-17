"""LaTeX insertion fragments with the revised manuscript's figure labels."""
from pathlib import Path

FIGURES = r'''% Requires graphicx; Tables 1--3 additionally require booktabs.
% Copy the generated figures/ directory alongside the manuscript .tex file.
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/ICASSP_Overview_Technical.pdf}
  \caption{Overview of the TV-budget controller.}
  \label{fig:overview}
\end{figure*}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/geometry.pdf}
  \caption{Nonmonotone segment TV for $f=(0.1,0.9)$ repeated in RGB.
  Proposal $(0.9,0.1)$ has a feasible endpoint but an infeasible middle;
  $(0.7,0.3)$ has a feasible prefix supporting bisection.}
  \label{fig:geometry}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/visual_single_column.pdf}
  \caption{Budget-controlled outputs for Kodak kodim23 at $\sigma=25/255$,
  using the same observation and DRUNet proposal. Full resized images use
  identical display scaling.}
  \label{fig:visual}
\end{figure}

\begin{figure}[t]
  \centering
  \begin{minipage}[t]{0.49\columnwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/test_robustness.pdf}
    \par\small (a) Denoising mismatch.
  \end{minipage}\hfill
  \begin{minipage}[t]{0.49\columnwidth}
    \centering
    \includegraphics[width=\linewidth]{figures/test_tradeoff.pdf}
    \par\small (b) Detection--noise trade-off.
  \end{minipage}
  \caption{Weak-structure detection on 100 held-out test geometries.
  (a) Mean AP versus noise-map multiplier $a$; shading indicates
  geometry-bootstrap 95\% CIs. (b) Complete fixed-blend and TV-budget scans,
  averaged over noise levels, repeats, and mismatch conditions. Stars mark
  validation-selected shared operating points; $\beta=0.2$ is also labeled.}
  \label{fig:weak_structure}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/precision.pdf}
  \caption{Precision--cost trade-off relative to 28-step bisection across
  four budgets. Time is the ratio of mean condition-median times; quality
  is the maximum per-image absolute PSNR difference across the same
  conditions. Kodak labels denote $255\sigma$.}
  \label{fig:precision}
\end{figure}
'''

SECTION = r'''\subsection{Weak-Structure Detectability under Denoising Mismatch}
\label{sec:weak_structure}
Beyond restoration fidelity, we ask whether budget control can recover
weak structures suppressed by denoising. We generate 40 validation and
100 test geometries containing bright, antialiased lines of width 1--3
pixels and contrast 0.03--0.12 in $256\times256$ images. Grayscale images
are replicated to RGB and corrupted by independent channel-wise Gaussian
noise at $\sigma\in\{15,25\}/255$, with three repeats per level. DRUNet
receives noise maps $a\sigma$, where $a\in\{1,1.5,2\}$. A fixed Sato
detector at scales $(1,2,3)$ is scored by per-image average precision
(AP) against geometric masks. Here, validation macro AP across all
conditions selects one shared coefficient $\eta=0.76$ for unconstrained
fixed blending and one shared budget $\beta=0.8$ for TV control, both
frozen for testing.

Figure~\ref{fig:weak_structure}(a) shows mismatch-dependent recovery:
AP changes relative to the proposal are $-0.0063$, $+0.0383$, and
$+0.0974$ for $a=1$, $1.5$, and $2$, respectively. Overall, TV control
raises test macro AP from $0.6070$ to $0.6501$, a gain of $0.0431$
(95\% CI $[0.0397,0.0467]$; 10,000 bootstrap resamples of clean geometries
retaining all paired conditions). At validation-selected operating points
(Fig.~\ref{fig:weak_structure}(b)), TV control nearly matches fixed
blending in AP ($0.6501$ versus $0.6502$) with a lower background RMSE
ratio ($0.1975$ versus $0.2410$, relative to the noisy observation).
'''


def write_fragments(output):
    folder = Path(output)/'latex'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'figures.tex').write_text(FIGURES)
    (folder/'weak_structure_section.tex').write_text(SECTION)
