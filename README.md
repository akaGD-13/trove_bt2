{\rtf1\ansi\ansicpg1252\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # trove_bt2\
\
\
\uc0\u25512 \u27874 \u21161 \u28572 \u20013 \u26399 \u25321 \u26102 \
\
\uc0\u22825 \u22320 \u26495 \u25968 \u25454 \u26242 \u26102 \u26080 \u27861 \u25552 \u21462 \
\
V1\
\
    \uc0\u25552 \u21462 \u28072 \u36300 \u24133  \u24471 \u20986 \u28072 \u20572 \u27604 \u29575 \u19982 \u36300 \u20572 \u27604 \u29575  9.5%\u20026 \u20998 \u30028 \
    \uc0\u36890 \u36807 \u27604 \u29575 \u25321 \u26102 \
V2\
\
    \uc0\u28072 \u36300 \u20572 \u27604 \u29575 \u21098 \u20992 \u24046 \u12289 \u36830 \u26495 \u27604 \u29575 \u21098 \u20992 \u24046 \u12289 \u22320 \u22825 \u26495 \u19982 \u22825 \u22320 \u26495 \u27604 \u29575 \u21098 \u20992 \u24046 \
    \uc0\u38656 \u35201 \u20445 \u23384 \u21069 \u19968 \u22825 \u30340 \u28072 \u20572 \u19982 \u36300 \u20572 \u30340 \u32929 \u31080 \
    \uc0\u19982 \u20170 \u22825 \u30340 \u21462 \u20132 \u38598  \u24471 \u20986 \u36830 \u32493 \u28072 \u20572 \u27604 \u29575 \u21644 \u36830 \u32493 \u36300 \u20572 \u27604 \u29575 \
    \uc0\u28072 \u36300 \u20572 \u27604 \u29575 \u21098 \u20992 \u24046  = \u28072 \u20572 \u27604 \u29575  - \u36300 \u20572 \u27604 \u29575 \
    \uc0\u36830 \u26495 \u27604 \u29575 \u21098 \u20992 \u24046  = \u36830 \u32493 \u28072 \u20572 \u27604 \u29575  - \u36830 \u32493 \u36300 \u20572 \u27604 \u29575 \
     \
V3\
\
     \uc0\u33258 \u30001 \u27969 \u36890 \u24066 \u20540 \u21152 \u26435 \u28072 \u36300 \u20572 \u27604 \u29575 \u21098 \u20992 \u24046 \
     \uc0\u33258 \u30001 \u27969 \u36890 \u24066 \u20540 \u21152 \u26435 \u36830 \u26495 \u27604 \u29575 \u21098 \u20992 \u24046  \
     \
\uc0\u27818 \u28145 300\
\
# notes.py:\
\
  gathering data of \uc0\u27818 \u28145 300, and calculating indexs:\
  \
      tradedate['\uc0\u28072 \u20572 \u27604 \u29575 '] = '' #1\
      tradedate['\uc0\u36300 \u20572 \u27604 \u29575 '] = '' #2\
      tradedate['\uc0\u22320 \u22825 \u26495 \u27604 \u29575 '] = '' #3\
      tradedate['\uc0\u22825 \u22320 \u26495 \u27604 \u29575 '] = '' #4\
      tradedate['\uc0\u22825 \u22320 \u26495 \u27604 \u29575 \u21098 \u20992 \u24046 '] = '' #5\
      tradedate['\uc0\u28072 \u36300 \u20572 \u27604 \u29575 \u21098 \u20992 \u24046 '] = '' #6\
      tradedate['\uc0\u36830 \u26495 \u27604 \u29575 \u21098 \u20992 \u24046 '] = '' #7\
      tradedate['\uc0\u33258 \u30001 \u27969 \u36890 \u24066 \u20540 \u21152 \u26435 \u28072 \u36300 \u20572 \u27604 \u29575 \u21098 \u20992 \u24046 '] = '' #8\
      tradedate['\uc0\u33258 \u30001 \u27969 \u36890 \u24066 \u20540 \u21152 \u26435 \u36830 \u26495 \u27604 \u29575 \u21098 \u20992 \u24046 '] = '' #9\
      tradedate['\uc0\u33258 \u30001 \u27969 \u36890 \u24066 \u20540 \u21152 \u26435 \u22320 \u22825 \u19982 \u22825 \u22320 \u26495 \u27604 \u29575 \u21098 \u20992 \u24046 '] = '' #10\
\
# bt_1.py\
  \uc0\u24403 \u25351 \u26631 \u22823 \u20110 x, \u20570 \u22810 \u12290 \u23567 \u20110 y\u65292 \u20570 \u31354 \
  \
        \uc0\u36816 \u34892 \u26102 \u38388 \u65306 \u21462 \u20915 \u20110  maxi, step \u30340 \u20540 \
        standard= '\uc0\u28072 \u36300 \u20572 \u27604 \u29575 \u21098 \u20992 \u24046 '\
        standard = '\uc0\u36830 \u26495 \u27604 \u29575 \u21098 \u20992 \u24046 '\
        standard = '\uc0\u33258 \u30001 \u27969 \u36890 \u24066 \u20540 \u21152 \u26435 \u28072 \u36300 \u20572 \u27604 \u29575 \u21098 \u20992 \u24046 '\
        standard = '\uc0\u33258 \u30001 \u27969 \u36890 \u24066 \u20540 \u21152 \u26435 \u36830 \u26495 \u27604 \u29575 \u21098 \u20992 \u24046 '\
    \uc0\u22235 \u20010 \u22270 \u65306 \u20197  1.png \u32467 \u23614 \
\
# bt_2.py:\
\uc0\u25253 \u21578 \u21407 \u25991 \
  \
    https://pdf.dfcfw.com/pdf/H3_AP202011181430494641_1.pdf?1605686153000.pdf  \
\uc0\u21482 \u22312 HMA30/HMA100>1.15\u19988 HMA30\u21644 HMA100\u37117 \u22823 \u20110 0\u30340 \u24773 \u24418 \u19979 \u25345 \u26377 \u22810 \u20179 \u65292 \u21097 \u19979 \u31354 \u20179 \
\
HMA\uc0\u35745 \u31639 \u26041 \u27861 \u65306 \
  \
    https://school.stockcharts.com/doku.php?id=technical_indicators:hull_moving_average\
\uc0\u22235 \u20010 \u22270 \u65306 \u20197  2.png \u32467 \u23614 }