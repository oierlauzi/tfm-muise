%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Class made according to the guidelines of the TFTs of the ETSIT-UPM
%
% Author:
% Oier Lauzirika Zarrabeitia (oier.lauzirika.zarrabeitia@alumnos.upm.es)
%
% Based on:
% https://www.overleaf.com/latex/templates/upm-report-template/sxtmzsytgthp
%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{etsit-report}
\RequirePackage[utf8]{inputenc}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load base class (memoir), passing Spanish as a secondary language to
% babel
%
\PassOptionsToPackage{main=english, spanish}{babel}
\LoadClass[12pt, a4paper, twoside, extrafontsizes]{memoir}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Base requirements
%
\RequirePackage{ifthen}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Degree
%
\makeatletter
\newcommand{\degree}[1]{\gdef\@degree{#1}}%
\newcommand{\@degree}{\@latex@warning@no@line{No \noexpand\degree given}}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Thesis type
%
\makeatletter
\newcommand{\thesistype}[1]{\gdef\@thesistype{#1}}%
\newcommand{\@thesistype}{\@latex@warning@no@line{No \noexpand\thesistype given}}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Thesis type
%
\makeatletter
\newcommand{\department}[1]{\gdef\@department{#1}}%
\newcommand{\@department}{\@latex@warning@no@line{No \noexpand\department given}}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Director
%
\makeatletter
\newcommand{\director}[1]{\gdef\@director{#1}}%
\newcommand{\@director}{\@latex@warning@no@line{No \noexpand\director given}}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Colours and background
%
\RequirePackage{xcolor}
\RequirePackage{pagecolor}
\RequirePackage{afterpage}
\RequirePackage{eso-pic}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figures
%
\RequirePackage{float}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{graphicx}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Licence
%
\RequirePackage[type={CC}, modifier={by-sa}, version={4.0}]{doclicense}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Language
%
\RequirePackage{babel}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% References
%
\RequirePackage[backend=biber,style=ieee]{biblatex}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Glossary
%
\RequirePackage[noredefwarn]{glossaries}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PDF utils
%
\RequirePackage{pdflscape}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multiline cells in tabular
%
\RequirePackage{makecell}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Margins
%
\RequirePackage[
   a4paper,
   twoside,
   inner=2cm, %Como maximo 2cm
   outer=2.5cm, %Como maximo 4cm
   top=2.5cm,
   bottom=2.5cm
]{geometry}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Math
%
\RequirePackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}
\RequirePackage{siunitx}
\RequirePackage{eurosym}
%\RequirePackage{physics}
\RequirePackage{cancel}

\numberwithin{equation}{chapter}
\DeclareSIUnit\angstrom{\text{Å}} % To avoid warning

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Source code
%
\RequirePackage{listings}
\RequirePackage{verbatim}

\lstdefinestyle{mystyle}{
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}

\lstset{style=mystyle}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constants
%
\def\universityname{Universidad Politécnica de Madrid}
\def\universityabbr{UPM}
\def\universitylogo{etsit-report/logos/upm-color.pdf}
\def\schoolname{Escuela Técnica Superior de Ingenieros de Telecomunicación}
\def\schoolabbr{ETSIT}
\def\schoollogo{etsit-report/logos/etsit-color.pdf}
\def\schoollogobw{etsit-report/logos/etsit-bw.pdf}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Colours
%
\definecolor{schoolcolor}{HTML}{f18e00}
\definecolor{schoolcolor2}{HTML}{004489}
\definecolor{linkcolor}{RGB}{0,170,230}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Headers and fonts
%
\def\hffont{\color{schoolcolor}\sffamily\small}
\makepagestyle{schoolheader}
\makeheadrule{schoolheader}{\textwidth}{\normalrulethickness}
\makeevenhead{schoolheader}{\hffont\thepage}{}{\hffont\leftmark}
\makeoddhead{schoolheader}{\hffont\rightmark}{}{\hffont\thepage}
\makeevenfoot{schoolheader}{}{}{}
\makeoddfoot{schoolheader}{}{}{}
\makepsmarks{schoolheader}{
    \nouppercaseheads
    \createmark{chapter}{both}{shownumber}{}{\space}
    \createmark{section}{right}{shownumber}{}{\space}
    \createplainmark{toc}{both}{\contentsname}
    \createplainmark{lof}{both}{\listfigurename}
    \createplainmark{lot}{both}{\listtablename}
    \createplainmark{bib}{both}{\bibname}
    \createplainmark{index}{both}{\indexname}
    \createplainmark{glossary}{both}{\glossaryname}
}
\pagestyle{schoolheader}
%\copypagestyle{cleared}{schoolheader}
%\makeevenhead{cleared}{\hffont\thepage}{}{}
\makeevenfoot{plain}{}{}{}
\makeoddfoot{plain}{}{}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hyperlinks
%
\RequirePackage[hidelinks,naturalnames]{hyperref}
\makeatletter
\hypersetup{
    pdfauthor={\@author},
    pdftitle={\@title},
    pdfcreator={\@author},
    breaklinks=true,
    linktoc=all,
    plainpages=false,
    unicode=true,
    %colorlinks=true,
    %linkcolor={linkcolor},
}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Spacing
%
\setlength{\parskip}{1em}
\linespread{1.2}
\setlength{\parindent}{0pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chapter, section and paragraph styles
%

% Chapeter
\makechapterstyle{etsitstyle}{%
    \renewcommand{\chapnamefont}{\normalfont\sffamily}
    \renewcommand{\chapnumfont}{
        \sffamily\fontsize{38}{38}\color{schoolcolor}\selectfont
    }
    \renewcommand{\chaptitlefont}{
        \HUGE\sffamily\raggedleft\color{black}\selectfont
    }
    \renewcommand{\printchaptername}{}
    \renewcommand{\chapternamenum}{}
    \renewcommand{\printchapternum}{\chapnumfont \thechapter}
    \renewcommand{\afterchapternum}{.\hfill}
    \renewcommand{\afterchaptertitle}{\par\nobreak\medskip\hrule\vskip\afterchapskip}
    
}
\chapterstyle{etsitstyle}


% TODO Intentar buscar unos estilos con un poco más de gusto
\setsecheadstyle              {\huge\sffamily\raggedright}
\setsubsecheadstyle           {\LARGE\sffamily\raggedright}
\setsubsubsecheadstyle        {\Large\sffamily\raggedright}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frontpage
%
\makeatletter
\newcommand{\makefrontpage}{%
    % Setup the page style
    \thispagestyle{empty}
    \newpagecolor{schoolcolor}\afterpage{\restorepagecolor}
    
    % Content
    \begin{center}
        \sffamily
        {\LARGE \MakeUppercase{\textbf{\universityname}}}\\*[2em]
        {\large \MakeUppercase{\textbf{\schoolname}}}\\*[2em]
        \includegraphics[width=6cm]{\schoollogobw}\\*[2em]
        {\LARGE \MakeUppercase{\textbf{\@degree}}}\\*[2em]
        {\LARGE \MakeUppercase{\@thesistype}}
        \vfill
        {\LARGE \@title}
        \vfill
        {\LARGE \MakeUppercase{\theauthor}}\\*[2em]
        {\LARGE \MakeUppercase{\the\year}}\\
    \end{center}
    \newpage
}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title
%
\makeatletter
\def\@maketitle{%
    % Setup the page style
    \thispagestyle{empty}
    
    % Content
    \begin{center}
        \sffamily
        {\LARGE \MakeUppercase{\textbf{\universityname}}}\\*[2em]
        {\large \MakeUppercase{\textbf{\schoolname}}}\\*[2em]
        \includegraphics[width=6cm]{\schoollogo}\\*[2em]
        {\LARGE \MakeUppercase{\textbf{\@degree}}}\\*[2em]
        {\LARGE \MakeUppercase{\@thesistype}}
        \vfill
        {\LARGE \@title}
        \vfill
        {\LARGE \MakeUppercase{\theauthor}}\\*[2em]
        {\LARGE \MakeUppercase{\the\year}}\\
    \end{center}
    \newpage
}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Information page
%
\makeatletter
\newcommand{\makeinfopage}{%
    \cleardoublepage
    \vspace*{\fill}
    
    % Setup the page style
    \thispagestyle{empty}
    
    % Content
    \colorbox{schoolcolor}{\parbox{\textwidth}{\raggedright\Large \MakeUppercase{\@degree}}}
    
    \colorbox{schoolcolor}{\parbox{\textwidth}{\raggedright\large \MakeUppercase{\@thesistype}}}
    \begin{tabular}{p{.2\textwidth} p{.7\textwidth}}
        \textbf{Título:}        & \@title \\[1em]
        \textbf{Autor:}         & \makecell[l]{\@author} \\[1em]
        \textbf{Tutor:}         & \makecell[l]{\@director} \\[1em]
        \textbf{Departamento:}  & \@department \\[1em]
    \end{tabular}
    
    \colorbox{schoolcolor}{\parbox{\textwidth}{\raggedright\large \MakeUppercase{Miembros del Tribunal}}}
    \begin{tabular}{p{.2\textwidth} p{.7\textwidth}}
        \textbf{Presidente:}    & D. \\[1em]
        \textbf{Vocal}          & D. \\[1em]
        \textbf{Secretario:}    & D. \\[1em]
        \textbf{Suplente:}      & D. \\[1em]
    \end{tabular}

    \vspace{2cm}
    
    Los miembros del tribunal acuerdan otorgar una calificación de:
    \vspace{1cm}
    \begin{flushright}
        Madrid, a \hspace{2em} de \hspace{6em} de 2024
    \end{flushright}

    \vspace*{\fill}
    \newpage
}
\makeatother
