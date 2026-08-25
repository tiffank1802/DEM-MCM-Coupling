% ============================================================
% MemoireMaster.cls — Classe LaTeX pour le Mémoire de MASTER
% (adaptée de RapportStage.cls pour respecter les règles de
%  présentation du mémoire de Master Mécanique — CentraleLyon
%  ENISE / Mines Saint-Étienne / Université de Lyon)
%
% Compilation : xelatex (ou lualatex) — nécessaire pour la police Garamond
%   xelatex memoire.tex
%   xelatex memoire.tex   (2 passages pour la table des matières)
% ============================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{MemoireMaster}[2026/08/03 v1.0 Classe pour le memoire de Master]

% ------------------------------------------------------------
% Options de la classe
% ------------------------------------------------------------
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[a4paper,12pt]{article}

% ------------------------------------------------------------
% Paquets essentiels - compatibilite moteur (pdfLaTeX, XeLaTeX, LuaLaTeX)
% ------------------------------------------------------------
\RequirePackage{ifxetex,ifluatex}
\newif\ifxetexorluatex
\ifxetex
  \xetexorluatextrue
\else
  \ifluatex
    \xetexorluatextrue
  \else
    \xetexorluatexfalse
  \fi
\fi

\ifxetexorluatex
  \RequirePackage{fontspec}
\else
  \RequirePackage[T1]{fontenc}
  \RequirePackage[utf8]{inputenc}
  \RequirePackage{lmodern}
\fi
\RequirePackage[french]{babel}

\ifxetexorluatex
  \IfFileExists{/usr/share/fonts/truetype/ebgaramond/EBGaramond12-Regular.ttf}{
    \setmainfont{EBGaramond12}[
      Path=/usr/share/fonts/truetype/ebgaramond/,
      Extension=.ttf,
      UprightFont=*-Regular,
      BoldFont=*-Bold,
      ItalicFont=*-Italic,
      Ligatures=TeX
    ]
  }{}
\else
  \IfFileExists{ebgaramond.sty}{\RequirePackage{ebgaramond}}{}
\fi

% Géométrie
\RequirePackage[
  top=2.5cm,
  bottom=2.5cm,
  left=2.5cm,
  right=2.5cm,
  headheight=20pt
]{geometry}

% Mathématiques
\RequirePackage{amsmath, amssymb, mathtools}

% Unités SI
\RequirePackage{siunitx}

% Graphiques
\RequirePackage{graphicx}
\RequirePackage{subcaption}
\RequirePackage[justification=centering]{caption}
\RequirePackage{xcolor}

% Titres et en-têtes
\RequirePackage{titlesec}
\RequirePackage{fancyhdr}

% Boîtes colorées
\RequirePackage[most]{tcolorbox}

% Bibliographie — thebibliography standard (pas besoin de biber/biblatex)
% Les références sont ajoutées manuellement avec \bibitem dans le document.

% Hyperliens
\RequirePackage[hidelinks]{hyperref}
\RequirePackage[open]{bookmark}

% Divers
\RequirePackage{url}
\RequirePackage{setspace}
\RequirePackage{enumitem}
\RequirePackage{csquotes}

% ============================================================
% COULEURS DU TEMPLATE
% ============================================================
\definecolor{minesbleu}{HTML}{4376AC}
\definecolor{minesbleuFonce}{HTML}{17497C}
\definecolor{minesGris}{HTML}{E3E6EA}
\definecolor{minesOrange}{HTML}{FF8C00}

% ============================================================
% CONFIGURATION DES LIENS HYPERTEXTES
% ============================================================
\hypersetup{
  colorlinks=true,
  linkcolor=minesbleuFonce,
  urlcolor=minesbleu,
  citecolor=minesOrange,
}

% ============================================================
% INTERLIGNE
% ============================================================
\renewcommand{\baselinestretch}{1.15}

% ============================================================
% CONFIGURATION DES TITRES
% ============================================================
\titleformat{\section}
  {\color{minesbleuFonce}\normalfont\Large\bfseries}
  {\thesection}{1em}{}
\titleformat{\subsection}
  {\color{minesbleuFonce}\normalfont\large\bfseries}
  {\thesubsection}{1em}{}
\titleformat{\subsubsection}
  {\color{minesbleuFonce}\normalfont\normalsize\bfseries}
  {\thesubsubsection}{1em}{}

% ============================================================
% EN-TÊTE ET PIED DE PAGE
% ============================================================
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textcolor{minesbleuFonce}{\leftmark}}
\fancyhead[R]{\small\textcolor{minesbleuFonce}{\@titre}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

% Pages sans en-tête (page de garde, résumés, sommaire...)
\fancypagestyle{sansentete}{
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

% ============================================================
% COMMANDES PERSONNALISÉES — MÉTADONNÉES
% ============================================================
\newcommand{\@titre}{Titre du mémoire}
\newcommand{\@auteur}{Auteur}
\newcommand{\@parcours}{Parcours}
\newcommand{\@entreprise}{Entreprise}
\newcommand{\@laboratoire}{Laboratoire}
\newcommand{\@ecole}{École d'origine}
\newcommand{\@formation}{Formation}
\newcommand{\@tuteurLabo}{Tuteur laboratoire/entreprise}
\newcommand{\@tuteurEco}{Tuteur école}
\newcommand{\@dateSoutenance}{JJ/MM/AAAA}
\newcommand{\@jury}{}

\newcommand{\titre}[1]{\renewcommand{\@titre}{#1}}
\newcommand{\auteur}[1]{\renewcommand{\@auteur}{#1}}
\newcommand{\parcoursmaster}[1]{\renewcommand{\@parcours}{#1}}
\newcommand{\entreprise}[1]{\renewcommand{\@entreprise}{#1}}
\newcommand{\laboratoire}[1]{\renewcommand{\@laboratoire}{#1}}
\newcommand{\ecole}[1]{\renewcommand{\@ecole}{#1}}
\newcommand{\formation}[1]{\renewcommand{\@formation}{#1}}
\newcommand{\tuteurLabo}[1]{\renewcommand{\@tuteurLabo}{#1}}
\newcommand{\tuteurEco}[1]{\renewcommand{\@tuteurEco}{#1}}
\newcommand{\dateSoutenance}[1]{\renewcommand{\@dateSoutenance}{#1}}
\newcommand{\jury}[1]{\renewcommand{\@jury}{#1}}

% ============================================================
% COMMANDES COMPLÉMENTAIRES POUR LA PAGE DE GARDE OFFICIELLE
% (définies ici, dans le template, pour alléger main.tex :
%  le document ne fournit plus que les valeurs via \renewcommand)
% ============================================================

% --- Aide : insère un logo s'il existe, sinon un cadre texte ---
\newcommand{\logoouplaceholder}[2]{%
  \IfFileExists{#1}{%
    \includegraphics[height=1.5cm]{#1}%
  }{%
    \fbox{%
      \parbox[c][1.6cm][c]{2.8cm}{%
        \centering
        \small #2%
      }%
    }%
  }%
}

% --- Aide : figure provisoire (à remplacer ensuite par \includegraphics) ---
\newcommand{\figplaceholder}[2]{%
  \fbox{%
    \begin{minipage}[c][0.28\textheight][c]{0.92\linewidth}
      \centering
      \textbf{Figure à insérer}\\[0.3cm]
      \texttt{\detokenize{#1}}\\[0.2cm]
      \textit{#2}
    \end{minipage}%
  }%
}

% --- Métadonnées de la page de garde officielle ---
\newcommand{\parcoursMaster}{À indiquer}
\newcommand{\laboratoireStage}{Laboratoire d'accueil}
\newcommand{\adresseLaboratoire}{Adresse du laboratoire}
\newcommand{\entrepriseStage}{Entreprise d'accueil}
\newcommand{\adresseEntreprise}{Adresse de l'entreprise}
\newcommand{\encadrantLaboEntreprise}{Encadrant(e) laboratoire/entreprise}
\newcommand{\encadrantEcole}{Encadrant(e) école}
\newcommand{\juryPresident}{M./Mme ---}
\newcommand{\juryMembreUn}{M./Mme ---}
\newcommand{\juryMembreDeux}{M./Mme ---}
\newcommand{\dateDebut}{JJ/MM/AAAA}
\newcommand{\dateFin}{JJ/MM/AAAA}

% ============================================================
% PAGE DE GARDE — modèle officiel du mémoire de Master Mécanique
% (CentraleLyon ENISE / Mines Saint-Étienne / École CentraleLyon /
%  Université de Lyon)
% ============================================================
\newcommand{\fairepagedegarde}{%
  \begin{titlepage}
    \centering
    \vspace*{-0.8cm}

    % ------------------------------------------------------------
    % Logos : placez vos fichiers dans un dossier ./logos/
    % Vous pouvez remplacer les noms de fichiers par les vrais logos.
    % ------------------------------------------------------------
    \begin{minipage}{0.20\linewidth}
      \centering
      \logoouplaceholder{logos/centrale-lyon-enise.png}{Centrale Lyon\\ ENISE}
    \end{minipage}
    \begin{minipage}{0.20\linewidth}
      \centering
      \logoouplaceholder{logos/emse.png}{ENSM\\ Saint-Étienne}
    \end{minipage}%
    \begin{minipage}{0.20\linewidth}
      \centering
      \logoouplaceholder{logos/centrale-lyon.png}{Centrale Lyon\\ ENISE}
    \end{minipage}
    \hfill
    \begin{minipage}{0.20\linewidth}
      \centering
      \logoouplaceholder{logos/univ-lyon.png}{Mines\\ Albi}
    \end{minipage}%
    \hfill

    \vspace{1.2cm}

    {\Large\scshape Diplôme de Master\par}
    \vspace{0.2cm}
    {\large Mention \textbf{Mécanique}\par}
    \vspace{0.2cm}
    {\large Parcours : \parcoursMaster\par}

    \vspace{1.2cm}

    {\large\scshape Mémoire de MASTER\\
    présenté par\\[0.3cm]
    \textbf{\@auteur}\par}

    \vspace{1.1cm}

    \textcolor{minesbleu}{\rule{\linewidth}{0.5mm}}\\[0.7cm]

    {\huge\bfseries\textcolor{minesbleuFonce}{\@titre}\par}

    \vspace{0.7cm}
    \textcolor{minesbleu}{\rule{\linewidth}{0.5mm}}\\[0.9cm]

    {\large Mémoire soutenu le \@dateSoutenance\par}

    \vfill

    % ------------------------------------------------------------
    % École / Laboratoire / Entreprise
    % Supprimer ou adapter la mention inutile selon le stage.
    % ------------------------------------------------------------
    \begin{minipage}[t]{0.48\linewidth}
      \raggedright
      \textbf{École :}\\
      \@ecole\\[0.35cm]

      \textbf{Laboratoire :}\\
      \laboratoireStage\\
      \adresseLaboratoire\\[0.35cm]

      % \textbf{Entreprise :}\\
      % \entrepriseStage\\
      % \adresseEntreprise
    \end{minipage}%
    \hfill
    \begin{minipage}[t]{0.48\linewidth}
      \raggedleft
      \textbf{Encadrants :}\\[0.2cm]

      Encadrants laboratoire/entreprise :\\
      \encadrantLaboEntreprise\\[0.35cm]

      Encadrant école :\\
      \encadrantEcole
    \end{minipage}

    \vspace{1cm}

    % ------------------------------------------------------------
    % Jury
    % ------------------------------------------------------------
    \begin{center}
      \textbf{Jury}\\[0.3cm]
      \begin{tabular}{rl}
        Président(e) : & \juryPresident\\
        Membre :       & \juryMembreUn\\
        Membre :       & \juryMembreDeux
      \end{tabular}
    \end{center}

    \vfill

    {\small Centrale Lyon-ENISE, EMSE, Centrale Lyon, Université de Lyon\par}
  \end{titlepage}
}
% ============================================================
% REMERCIEMENTS (non comptés dans les 35 pages)
% ============================================================

\newcommand{\fairemerciements}[1]{
  \pagestyle{sansentete}
  \vspace*{1cm}
  \begin{center}
    {\Large\bfseries\textcolor{minesbleuFonce}{Remerciements}}
  \end{center}
  \vspace{1cm}
  #1
  \newpage
}
% ============================================================
% RÉSUMÉ / ABSTRACT (1 page maximum chacun — non comptés dans les 35 pages)
% ============================================================
\newenvironment{resumefr}{
  \newpage
  \pagestyle{sansentete}
  \begin{center}{\Large\bfseries\textcolor{minesbleuFonce}{Résumé}}\end{center}
  \vspace{0.5cm}
}{
  \newpage
}

\newenvironment{resumeen}{
  \newpage
  \pagestyle{sansentete}
  \begin{center}{\Large\bfseries\textcolor{minesbleuFonce}{Abstract}}\end{center}
  \vspace{0.5cm}
}{
  \newpage
}

% ============================================================
% TABLE DES MATIÈRES / LISTES (non comptées dans les 35 pages)
% ============================================================
\newcommand{\tabledematieres}{
  \pagestyle{sansentete}
  \tableofcontents
  \newpage
}

\newcommand{\listefiguresdoc}{
  \listoffigures
  \newpage
}

\newcommand{\listetableauxdoc}{
  \listoftables
  \newpage
}



% ============================================================
% DÉBUT DU CORPS COMPTABILISÉ (page 1 = introduction, règle des 35 pages)
% ============================================================
\newcommand{\debutcorpsmemoire}{
  \pagestyle{fancy}
  \setcounter{page}{1}
}

% ============================================================
% INSERTION DE FIGURES
% ============================================================
\newcommand{\insererfigure}[4]{
  \begin{figure}[ht]
    \centering
    \includegraphics[height=#2]{#1}
    \caption{#3}
    \label{fig:#4}
  \end{figure}
}

% ============================================================
% ENVIRONNEMENTS POUR LES ENCADRÉS
% ============================================================
\newtcolorbox{encadre}{
  colback=minesGris!30,
  colframe=minesbleu,
  fonttitle=\bfseries,
  title=Information
}

\newtcolorbox{pointcle}{
  colback=minesOrange!10,
  colframe=minesOrange,
  fonttitle=\bfseries,
  title=Point clé
}

\endinput