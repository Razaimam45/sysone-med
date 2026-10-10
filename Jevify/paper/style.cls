% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A new style for FAIR pre-prints
% Author: dlp@meta.com
% Version: 1.1
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Class declarations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{style}
\LoadClassWithOptions{article}

% Layout %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[top=2.25cm, bottom=2.5cm, left=2.5cm, right=2.5cm, columnsep=0.65cm]{geometry}
\RequirePackage{microtype}
\RequirePackage{placeins}
\RequirePackage{hyphenat}
\RequirePackage{setspace}
\RequirePackage{parskip}
\RequirePackage[latin, english]{babel}
\RequirePackage{lipsum}
\RequirePackage{etoolbox}

\ifPDFTeX\DisableLigatures[f]{family=sf*}\fi 

% Graphics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{graphicx}
\RequirePackage{subcaption}

% Tables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{booktabs}
\RequirePackage{nicematrix}
\RequirePackage{multirow}
\RequirePackage{bm}
\newcommand{\nm}[1]{#1}

% Colorful stuff %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[most]{tcolorbox}
\RequirePackage{xcolor}
\RequirePackage{tabularx}
\definecolor{meituanyellow}{HTML}{FFD100}
\definecolor{meituanlink}{HTML}{A86400}
\definecolor{metablue}{HTML}{0064E0}
\definecolor{metafg}{HTML}{1C2B33}
\definecolor{metabg}{HTML}{E8F5FD}

% References %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{hyperref}
\RequirePackage[noabbrev,nameinlink]{cleveref}
\hypersetup{
  colorlinks,
  linkcolor=metablue,
  citecolor=metablue,
  urlcolor=metablue
}

\RequirePackage[numbers,square,sort&compress]{natbib}
\def\bibfont{\small}

% Section and caption format %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{titlesec}
\titleformat*{\paragraph}{\itshape}
\titleformat*{\section}{\Large\sffamily\bfseries}
\titleformat*{\subsection}{\large\sffamily\bfseries}
\titleformat*{\subsubsection}{\normalsize\sffamily\bfseries}
\DeclareTextFontCommand{\textbf}{\bfseries\sffamily}

\RequirePackage{caption}
\DeclareCaptionLabelSeparator{custom}{}
\DeclareCaptionFormat{custom}{{\sffamily\textbf{#1 #2}} #3}
\captionsetup{singlelinecheck=false,format=custom,labelsep=custom,font=small}
\captionsetup[sub]{singlelinecheck=true,format=custom,labelsep=custom,font=small}

% Meta custom fonts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ifPDFTeX
\renewcommand*\ttdefault{cmvtt}
\RequirePackage[T1]{fontenc}
\DeclareFontFamily{T1}{optimistic}{}
\DeclareFontShape{T1}{optimistic}{m}{n}{<-> s * [0.88] assets/optimistic}{}
\DeclareFontShape{T1}{optimistic}{b}{n}{<-> s * [0.88] assets/optimistic}{}
\pdfmapline{+optimistic < assets/Optimistic.ttf <T1-WGL4.enc}
\renewcommand{\sfdefault}{optimistic}
\else
% Load the same bundled template font directly with XeTeX/Tectonic.
\RequirePackage{fontspec}
\setsansfont{Optimistic.ttf}[Path=assets/,Scale=0.88,BoldFont=Optimistic.ttf]
\fi

% Authors, affiliations, infos %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand\addtolist[5][]{
  \begingroup
    \if\relax#3\relax\def\sep{}\else\def\sep{#5}\fi
    \let\protect\@unexpandable@protect 
    \xdef#3{\expandafter{#3}\sep #4[#1]{#2}}%
  \endgroup
}

\newcommand\authorlist{}
\newcommand\authorformat[2][]{{\sffamily #2\textsuperscript{#1}}}
\renewcommand\author[2][]{\addtolist[#1]{#2}{\authorlist}{\authorformat}{, }}

\newcommand\affiliationlist{}
\newcommand\affiliationformat[2][]{{\normalsize \textsuperscript{#1}#2}}
\newcommand\affiliation[2][]{\addtolist[#1]{#2}{\affiliationlist}{\affiliationformat}{, }}

\newcommand\contributionlist{}
\newcommand\contributionformat[2][]{{\small \textsuperscript{#1}#2}}
\newcommand\contribution[2][]{\addtolist[#1]{#2}{\contributionlist}{\contributionformat}{, }}

\newcommand\metadatalist{}
\newcommand\metadataformat[2][]{\ifstrempty{#1}{{\small #2}}{{\small {\sffamily \bfseries #1:} #2}}}
\newcommand\metadatabarlist{}
\newcommand\metadatabarsep{\hfill\hspace{0.8em}{\color{metablue!35}\rule[-0.35ex]{0.6pt}{1.25em}}\hspace{0.8em}\hfill}
\newcommand\metadata[2][]{%
  \addtolist[#1]{#2}{\metadatalist}{\metadataformat}{\par}%
  \addtolist[#1]{#2}{\metadatabarlist}{\metadataformat}{\noexpand\metadatabarsep}%
}

\newcommand\resourcelist{}
\newcommand{\resourceicon}[1]{%
  \raisebox{-0.23\height}{#1}%
}
\newcommand{\resourceformat}[3]{%
  \resourceicon{#1} & {\sffamily\bfseries #2} &
  \href{#3}{\nolinkurl{#3}}\\%
}
\newcommand{\resource}[3]{%
  \g@addto@macro\resourcelist{\resourceformat{#1}{#2}{#3}}%
}
\newcommand{\printresources}{%
  \ifdefempty{\resourcelist}{}{%
    \vskip 0.15cm
    {\footnotesize
      \setlength{\tabcolsep}{4pt}%
      \renewcommand{\arraystretch}{1.10}%
      \begin{tabularx}{\linewidth}{@{}c>{\raggedright\arraybackslash}p{0.28\linewidth}>{\raggedright\arraybackslash}X@{}}
        \resourcelist
      \end{tabularx}%
    }%
  }%
}

\renewcommand{\abstract}[1]{\def\abstractlist{{\color{metafg} #1}}}
\newcommand{\email}[1]{\href{mailto:#1}{\texttt{#1}}}

\renewcommand\date[1]{\metadata[Date]{#1}}
\newcommand\correspondence[1]{\metadata[Correspondence]{#1}}

\renewcommand{\title}[1]{\newcommand{\titlelist}{{\huge\sffamily #1}}}
\newcommand{\titlelogoimage}{}
\newcommand{\titlelogo}[1]{\def\titlelogoimage{#1}}

\newcommand{\toplogoimage}{}
\newcommand{\toplogooptions}{}
\newcommand{\toplogo}[2][]{\def\toplogooptions{#1}\def\toplogoimage{#2}}
\newcommand{\printtoplogo}{%
  \ifdefempty{\toplogoimage}{}{%
    {\huge\sffamily\expandafter\includegraphics\expandafter[\toplogooptions,height=1.7em]{\toplogoimage}}\par
    \vskip 0.3cm
  }%
}

\newcommand{\printtitle}{%
  \printtoplogo
  \ifdefempty{\titlelogoimage}{%
    \setstretch{1.618}%
    \titlelist\par
  }{%
    \begin{minipage}[c]{0.075\linewidth}
      \centering\includegraphics[height=30pt]{\titlelogoimage}%
    \end{minipage}\hspace{0.015\linewidth}
    \begin{minipage}[c]{0.84\linewidth}
      \setstretch{1.35}%
      \titlelist\par
    \end{minipage}\par
  }%
}

\newcommand{\beginappendix}{\appendix{\huge\sffamily Appendix\par}}

% Titlebox %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\newcommand{\mymaketitle}{%
  \tcbset{enhanced,frame hidden}
  \tcbset{left=0.5cm}
  \tcbset{right=0.5cm}
  \tcbset{top=0.5cm}
  \tcbset{bottom=0.15cm}
  \tcbset{arc=10pt}
  \tcbset{colback=metabg}
  \tcbset{before skip=0pt}
  \tcbset{grow to left by=1.5pt}
  \tcbset{grow to right by=1.5pt}
  \tcbset{overlay={}}
  % 
  \begin{tcolorbox}
    \setlength{\parindent}{0cm}
    \setlength{\parskip}{0.5cm}
    {
      \setlength{\parskip}{0cm}
      \raggedright
      \nohyphens
      %
      {
        \printtitle
      }
      \vskip 0.2cm
      %
      \authorlist\par
      \vskip 0.2cm
      %
      \affiliationlist\par
      %
      \contributionlist\par
    }
    %
    \abstractlist\par
    \printresources
    \ifdefempty{\metadatabarlist}{}{%
      \setlength{\parskip}{0cm}%
      \vskip 0.15cm
      {\noindent\makebox[\linewidth][c]{\color{metablue!35}\rule{\dimexpr\linewidth+0.85cm\relax}{0.6pt}}\par}
      \vskip 0.12cm
      {\let\small\footnotesize\mbox{}\metadatabarlist\mbox{}\par}
    }
  \end{tcolorbox}
  \tcbset{reset}
  \FloatBarrier
}

\newif\iftitlecard
\newif\iftitlecompare
\newcommand{\abstracttablabel}{A\kern.14em B\kern.14em S\kern.14em T\kern.14em R\kern.14em A\kern.14em C\kern.14em T}
\newcommand{\mymaketitlecard}{%
  \ifdefempty{\toplogoimage}{}{%
    \begin{tikzpicture}[remember picture, overlay]
      \node[anchor=north west, inner sep=0pt] at ([xshift=2.5cm, yshift=-0.85cm]current page.north west)
        {\expandafter\includegraphics\expandafter[\toplogooptions,height=1.05cm]{\toplogoimage}};
    \end{tikzpicture}%
  }%
  {%
    \setlength{\parindent}{0cm}%
    \setlength{\parskip}{0cm}%
    \raggedright
    \let\printtoplogo\relax
    \nohyphens{\printtitle}
    \vskip 0.1cm
    {\fontsize{9.5pt}{12pt}\selectfont\authorlist\par}
    \vskip 0.1cm
    \affiliationlist\par
    \contributionlist\par
  }%
  \vskip 0.35cm
  \begin{tcolorbox}[enhanced, colback=metabg, colframe=metablue!75, boxrule=0.8pt, arc=6pt,
      left=0.5cm, right=0.5cm, top=0.2cm, bottom=0.3cm, before skip=0pt, middle=0.12cm, bottom=0.22cm,
      fonttitle=\sffamily\bfseries\footnotesize, coltitle=metablue!85!black,
      title={\abstracttablabel},
      attach boxed title to top left={xshift=0.5cm, yshift*=-\tcboxedtitleheight/2},
      boxed title style={colback=white, colframe=metablue!75, boxrule=0.8pt, arc=7pt, left=0.25cm, right=0.25cm,
        top=0.08cm, bottom=0.08cm},
      segmentation code={\draw[metablue!35, line width=0.6pt]
        ([xshift=0.12cm]segmentation.west) -- ([xshift=-0.12cm]segmentation.east);},
      bottomtitle=0pt, boxsep=0pt]
    \setlength{\parindent}{0cm}%
    \abstractlist\par
    \printresources
    \ifdefempty{\metadatabarlist}{}{%
      \tcblower
      \setlength{\parskip}{0cm}%
      {\let\small\footnotesize\mbox{}\metadatabarlist\mbox{}\par}
    }
  \end{tcolorbox}
  \FloatBarrier
}

\if@twocolumn%
\renewcommand{\maketitle}{
\twocolumn[%
  \mymaketitle
  \vskip 0.15cm
]
}%
\else%
\renewcommand{\maketitle}{
  \iftitlecompare\mymaketitlecard\clearpage\mymaketitle
  \else\iftitlecard\mymaketitlecard\else\mymaketitle\fi\fi
}%
\fi
