import base64
import zlib

import torch
from torch import nn
import torch.nn.functional as F


D = 16
H = 2
N = 2
WINDOW = 5
_WEIGHTS = "c-jq?^+OctAIB}uE-Vm1klL7@XJ!`I1?<8)m9s_hY){lP&+hI-lv<XJ>3L>$7qAn*=6UK_XPlmAo^_(0@%0b*e13V?YYNdpZ?BEAzojVIYkwSYGwX;h!TZ5aNjYivQrl|l<yH)UV~TTa|0Bi&R_dJ^rk)C9C$DlmwU4x23w+iy3u~3v3!G7o`A6%u%SJh(4Zb`zB|N1Hys_}sP|?X8ENklDQFb*@pFCK$*S-s1r57dlPM%;cXMc6<4}`KrdD8=n#V^`uTYcN{vW^vN@<w=?lI@bl8iVYSqKipn2Lu;Udz7BsM5Z#Rsw)Cd%6|zg<i<E3Cy&z7*`L|azz)4tMJ_O#zv^RCW|XB=B$u@@Fb3y-typAN4h}r<gdZ9p9pUVF?tA-jb`obxSy8bv*;h6LTM|sIsN~u+_vv~mt;-f0ol;sF8PuMN))gaC8m2me)InK!?dYiOR?d^L#9Xi-nT;8P#~WXL3&1?lNctOo4D?V4)>%>iP?_A%(k=F(^Cw&S!DV1MycN7^R3*<&neAO4Sr{~IjTpbZyK7Qe8{6oVvC1nDkvhrV@6ZM%^uVoSZ`JXRDtmvBX1{jyj9jlmOZS%9(}k2JcC)>U{j<H$zQn%K{y9arZ?zvtew~tM@0>C{xdZ>NU>4HRl>MqOK6g&5zIY9Jg>a+#JW~zg6zO)IkX`sftR7hpe=lWlv*No_r3He~1z_fB;_!GcA7pDWir^sPl%~|JW^X-@<!;0o5OhuvpU5SFKV3KE*qHakd1{);X_-#PDr<cG<wb20<R{1-s>FiPZSkSr)zUSL23w&SkH_|cv+h)6tdd0JQ1ygDWTo2PKZ)8@+Ll?N<X{4<j)tK1t)0OMs;e*`%>%<-W#&H+-r5~r1Qxla0DKYXd+(AxH9?s2NYT{&>JYh>8pf^g+;Xbk<=8HItvnqrp~lD|<Y8tSwY&6*&{8}pHjW!6B4Sd0D$giEWW9V^>7fR#tI$J4Kz!78c3d1emskgnLXUMD_o--t{Dj`4{>AQ9CL!VE42Fj{Oa<6?=pJwhMv=>qa#xjj0$x=gctb%I{J}5EZB(mmy0#Dwj;Txj6Ss+Mp#e~2ehJ%>>-E*(d*voPMMMfm$eQ?9buifz9t6q43w1X&TYvAmPkqoSBVMimU!_dysJ21=?5RK+pgqjb=`HdrEzw`)8CDb{K8H=2njizLD7;R+7S>>KpqsvciuHzg$FFD@+h3Sj+Chn@omh$9l;X(6%m;BAU4hm%z6Sn?d0^Zjej|GFgGB^cq<-|c2$D-DYRTthe;i8|M2PH3C}X9{3CyP7a{Gjy$TV#}9{TNm9eT5*rCu!VA?M>gsdNHTuZ{j>TMSUmk(CUuqTqj226>yuO$U5^#aj9wf-Zk1mr}KrW|n<%wmd+bBQuFD;9cTHWCl!6Bt1of+sv?=kuskZ>?uBvrwa^`X3F%p@Xl~wr`Bkv@kLS_(*~@&aL#M?0k=D^J^38AND3i7V*g2v=@iZ3`b7LAeW(40JXFi6-|2jE6Z5#Vk#7Rb7a|mzIfLp#h1!E0M(j~shBJOE(Zh6#yA3+Zb9fvZs4Vy7B!o#9l_<+_a73!b{GgXg2(<$IraoggP=k$))OqIz>jYxD`V*Ed)CUR1c;gW|gUV#gd`I@F){dzLU8cBzm2r?c(X)c@oa0~<Bvl;YJp?uhCUpsMtKc2;Gu@w`p|k>(as7Rgb`RRfnoK@jsf6ma;T$B6s3-qzdu`yv<JeEaP2`I7P+o1_B4jBpGMcU@<tUYVjuJHfL1nbtMdReh;4=S@I*+{{77HSjKfV7D3*DoHUlLA&w|IME7u8?st!>m>g3qXrnj`Ii4oH#-Vl9G1PQwjMJ~l}G09U%JQM#|nnj5=FstxvnC@?>9Myc7jg!c2U5gz-887~s@;l!9nL~Rg@#JJL^+_<6mI8#q%88?r<>Aay7ASaP9zAhi|O)g32kLrAr-NtW7S9&A!Bj8MR;Z$lI%oKcaZh%<E%301*<dJKKX(Z?7Y9<^pI(lCg9x&FocwW+gUgH`LN`#O2OF9jo$&@iSv;_)ep6EH!DZ;J4gW+NSV`2jL;AvsJvX@k-o^&&CN`69)V%mtq*j~mzMqJS|-5?)vU*zOOmvRb@!$0|FB8BoxAr!v~dJ^gMEb}(fnmE<t3B4GXg-rC>C<i*lE3nP<dfH{u;Y=lBSRWj&&VX;UQePM-rCKpls8+&r_kOK4(o|)HbNGK!P00Dy_zrm25HE#DP*Zs*vwRhP39nDiQEvOT<sFMWCbkTWG~dXx6(!=0l*5JV=yt+u<qe7vV-Z=rVHygz6|VEm!*(E>@eOT$g!|=vf*EY&UFO?zrjnx^#t)%m)l9*wJf_0k_e!psr;|-#5bH;@7rV;5Fim=kHr4#TI^tsWiu?urT(FCbz;DFXiM{3-qE7QxQFdZ9QUqM3Ylzxmj&J^JOeK<A<(^1CnCaSx%vKL#SNO3~fJ}frSqB=;ywMsf{pFJUe}OZ&Q5&EQMaCJ;ZJ|n*`U}i*e{Zz4CX`^@S)a?m>=`zT|LwO;{=V!Yb(((7{f>+QujI#;Tz^|QLmfz^Q7=j>iCCJGpWrpbS5jYsNAt-uNVK*&{twX^79cx9pZcEt(f9(rvf@6K*i~CDEmhCZ>osfXU~GHItb%a%E%O!jDsDu@^Xn-uQHwgR<hQ9$E|uRU_)C@%SG2g2DxyRgBAhn|-A|D`SYO{jnj}e&uqNoQKmvCc*z{avtxJ}s^8qme9lrlLGD!J>biu#0NxG!nWx}P}$XaS26$RpK`#}I(jZV{>AxE@}e7~l(u@9C%L38>gb6j1bGc@aI3=YXr)aJk{<D);U%&X5$_*d=a<y;Gt_t^jBJ?bpABejk^#twBSN%II1tP4Ej8N5QsDXb~evWI0n?x^YTsEq5qjrsbTxF!Cv^6-)nxJ+AQT2vb6Yaf4^!i3*#q;!)SPgMF>f@tKRKir7X(ZmRC8Qx83sP3f`+*hbSVqd{l?$i97cI%^Vu({eks+{UxaN7JLydUu~af4@^aj-Ctr%WaINreIyaUwm*^BbHmhs)2E1})k!N5KZtN_0x@3?FNi+Cs(0AIf{}n(cGY5TgJoUzLc$NccNB80#%6<SV%m-awimTKID%)!lo1trMH!jgqc2r{M{>iXE-4Q2XHT`Q1L++X}zLH-N|a5aF3CMo9(VX)C}8@~t_mcrFDIP3aW<KdTdo@>qe5V5pJObh^@79deWdSm1qWs9=Xv)KYtfW48Ui{gQpIeWv}SqrKy~{g6Gy4(-iTf3Ux`Yo`7s6FhNPw}cZWi%@r8p`4UBlZvp!OTiLE*&=@SZiOzbE>{J!<-^Rk<Qf@7GC*LP=p*1M-30!X@KK*7cV~OZOF)uw(iEoEa_5@9b#ub^e7Ja%s!Z%^)E4vDXeQ5mVfS&edW!~Z5Ub-!5-#W`6Ji5r{f)GTAeUI7$IFy*OUIe-n2)lLE~G`bMcs^_+1(rq19$OxVb$e9jTQw?akj`MCYM&--@xQnKdUChP1pBKrXB*(Fv%o&Rv2#?Ko8|#>y>hxR^amS$;27miXUdv)Zg|@P<+y3HJ`hxOUk*xaqAfW2<|*o<k3VE&aIwvb`(B%m+*Ipr$l|Efa=a)hU2jTzGJ3nce)Yf8ZBh9uj6>DTP>8&u->?U@?Xr>VojTmgniS0FBMd(!)L0UxFb(+v4Ly+$LcFN8~KItDDVBXlxcX8&x&6LZ?%~^Zr+D2C;H-lT2j3ar3%;_%PLNmC3?NG2448KDHFkUuDVtatK>)HklxoOTe;|G#6)$ytqGQ;-f%Vqg-kAyB}U3^=>{Z6D83`imTyc2<r>N_<}1`%8i6mA9AY;8FLf9WVt;D)o#ny)D?Xk~)V7PCko4RbWu5j0`IFj6Lf1bDe=3!h>C`EuPzls{6gc9a7=D8<0u-!~`&euU6{9uy-W&;k!=pTD-oMB_!c~)s9G4$)_sPb58D9Xs+-mX<eqG&T*c&jPYt6Qa6$CvnL4L=dVh7?G>?~oYehir`PsQ9NN6{2xgkY#eEY9D>=bL78XXJI@IQdcAE>HE0m;YcZsS3^HzeIjSGGI-nkI)4sV3ksH<RUCVnsd#xJMJK}3lHVisdr#R%4)rlLBk!`XZaFR-L}mfG<DI&ihruT+kPSX7@di+>}K_>=XW|>P13I6SBy52;ro@%^u>aGm>4(M{NB`w?YX~K=^g)UbfbAUzg=@<c9$X3AXLp%*ZE38F$gL&y9J)sC#c7XEUjzIU1v+8T;PbdYV(pbVkh~#9-(B*F?17cs<XG6p(5}nBH6p%HKgb*+^NKoPS62ALYnj+_!ijO*v^GD<3;@`NG|Ow#(>1c3!F*5<U9dJ>DPcsX{x+qCxC2IGh+sHOD#?36MG2{g_YbC!hv<wZh$&Usq(5;fZR;&V9c;Nd<H&Z3$*&e$pXMlN#cy(O{v5Hbp#SF&xJ+gAXiu3W<)6Kkc%`F;+T*m0REI`YYCq5)BzTEW)ZE0BKX;~0&S0efi);66-pHoKT|ehJ{)G;#yr?W@KW~)1Hd*Iknp%oN*>=#d$p@3dQ_W77nA7;Wymmbwb?^hN*j~K^!fPqmiO2jGTk&ue#@^C=PIAow@jJ`=YJ$?P&e`I*j#vys4k3caXMkX*v&FTYDohkhEIWhHO1Lmtfwv2=HlH+TJ+&#z3+s!U`UiA6)CUL?Uqw;s?r!*i4<YHx>Vf&!mL+)J7F&QukcJ6=v&J)vc!RYN~G&@{W5wm_C<d{^kwG|)zBaG0KW}J$RpTi{2X!v-=?HGI>=H7!;Fo}WWEjdy)q!CB{4Lz9taU%Da(m~moo2*ETi^IbHRD7QfR<$Z*p2)gY4nvdgc+WV7+{dX@zYgZeTTIHP0eAN2xAdC%YLns1V;qVj&Z$5Na6AB9F-<<sT(eV2hhWR|`(JsAnsG(M95FZDdJ?utv?(L*WIF#D*w{rP345Ka`q-_qr|dXYfd^ClJCa-{RulTsFTE#t<)%U#X9HH+daBSG|JVCpfH?X13Ke&f=q$`g#X?m^>FfCKp;$6KCZ=lxwI;VpM51vRPi`4gou<&t$1L(Y75~?Ku?tG;TV1MX9bOpg+lfGCpgCI|HS`N?R{|KKaocZyw=(gTA0A!rE>@j<@0%K<CQmlw-(5Z!4*r+Fi@UQut)_i|`6{2wnNQ(WR=68Sz%FrGCh=T#ttHi7UGX2%&p6`7tI(ti%fW{zATn(_T*lp=aVqc_fGs7OQpW6M{i*ltt=4Zzg8YE7a}aN~}f2<fig{p(gl;>V}7F7BU?vq_T;>)kC|NspE-QD`~E>93bCG2X?1X&9J59_mUZ=m^$lS;@bP35sA@bqi2B5+Rvs<(Y38fN(OSFC@hg9=eI-XOMDuPu>P%_B7PDtiW^`Bu^szRyej6MG?G{z|Crxk`-MuOmyy$i7SY|b5Y!45U?N@0ErYKGntVcBjK5ntPkf0LfLp$)I;S5rKT->kZqAyD7su6h@UvKu--X%^^PQKdj^x*b?O+-Lw8KPWtJB@Z@)gg)$BP-*B0ZOj$Bt@sL{_Z>{y-ape?7lw&6q1{7Pt&96}M0h%1&n^TcsE;uPI>EhO(rL)$US>F@H#RAV<uEzoKu<^?|fM!d;y?W$}92u+KDKTzyT%e^w@fKQK2L2OXl}yh2;>){I?UklT!I&AfyT`m2%&|Ay<Vb;y2VbApvuP|4Uw>@xmBoe}$i9AX-dTq4Gh34DsVGZ?1TXTP&Ly^CO`+TT{9k0bj@JlU1l#adw_`U8PNT-X73P>kno%re`*C`5kddipL&DmB_Eny!|#;dLxYtP}ZH+*`Z`RgP81=MsOKv(XypK=f2$wo=u2l59~Vcc;`oj{iB1r~d6Y<~Wvm)KTTAbWC-$Os$qy>1ddC*}k_V39Z2AYFW-+$|`FQ<WJ{8Vj+r)HuFgl6MM%FfS+vZoU>9|1Ox2&q=SL(Nu3UMu&uT2kZUmym@OqvWnrLS;EjI}ce-RA9+H%u(v;ek*o&-R=198U?m1#N9@-oFlN_zfCIvIum|#QQ69n33EttZ2R-}|9e`4(X7u#4p!;oN~ip%ntvU*&s`F=1zg=RjJtK!RoBZ4z+_sgcE>9#Gw8cDTS%Gwf^;BS<h!AWUNY#04Ad_Pgg4Jx^Jkha<QD#V@CThAh|kOS?{`~#`m#yX>F>qoeZ$<P+C3$>>y^L$zQdfzX4Xi^>F5_>E#+Omy{7rjX{{4q(h%Hs~m_UyoRyT|r~og#)Y<LbO;9txZ3Vd_Zbvihsw4K(ziF|#ahuq*OvrUl&jEi(~dv`13^qFs<zH1B$O1X!#-0JYc(c@by@f0ieES7HOGLRf72CoYGtB5UcRnA5~?M%ODM@;uAE$;5qs3#Ju+7hD2m#B*$IxURo1=>bn8R>VnnidUh<dO$#pd|01pfGo$e<#2x<HI2AaR5!7%P}@7t^DuU=vQv4&T=V^E+NbqEW$7bD$M1%pqP~((qcupO5z=C<+~p-yWi;CX?c+<;suE|a3D8V$v&29f@{u~DjP%ruIZsuf8}Si%w)-SD2JF<vSht#b^Ns8)gQKa$q(FdUw2BM#Cz=gJmUs3H*6S!An9y=t8BF5hTPR<s_qK`hDegjY!(giI0huNCiM>W1gd1$5^jUGywvn2SjW#+ey~?I>gM6E9xUFvTTJZ-qRqzGZ3Y*HWDa)kB$z@=1;7yXt_CN3sycx8?U$}aKO|BYxX5cyZg?SIx+0uw7AQx^)n+?vgGo@-;_rMu)X%Mw-H3rB@F>^rq))(FgTOG25r1`b+9;o{}6N!XEKQqFyqtLL0i!Xy?O_=#IHQ2C&C~IzDlgAqfPufrss|{o?n@MGvEng~;4^g4Y_i~V!Ys)7m=~I3C5<~P!Gy@jtRdii#mZ=AENZ(AgX98N0($_Of?Vp$;XEPPvcw&^)!(8%BFO#K}ZzH=O4j?o1i&zm182+fq@&f#d^`5d4*$3O(+?Iw?yp{up>3;Qr@|J9it|FV0kL0gOZ?Hw~soXoXjnP1TuO*@`XGgj^#^bD>11+FBy9yscIhhQB(4x>&unJ@|iT=Np=F}#!uWl90L??ME`WTUfNNp5Ro89Pn)h?dTlh2YZtSQQ8q%7jF{-sT_2|$&n6o)Mu7-~zdK8ObAW6~n_@R!uJo>;Xst~HIwWy*EN&7|WW{B<lH&;s$g;)(l@e2H#UP0CGwYiEftQF;!m0Uk+~7f>}#(bhL=17aGUDGt(0v7flf@+jWV4Ukt@4SBZ~9T}<}&!3(+UcQgtc9W!2`<xKXPtkiozdYA`*t$SHhO~o)!T{u7av~azS7CQt9;H#_JLjIjIod97<706*(!-WuKBW&e?*U_!0YCy>@rwmrJ(nysh@a8Zh$H@9Y$UlrI<4k2>!r)cg9H|4Avb~F^epCA{jKz#jb+}#K9<44b1<9uY#IlbBRXFY9|30)U8r-y4C#sX-aU(0V>#n&ZmCncjrgJIAL=FdkKi!cp>(Rs63byVt&6#)|L1*9Hri6bzvnuj0NJ3l5$*$*c$55S+<+Mbrq}jG63e3YmNX!$)Uo7UrXnGR`CA~d!&*cAlk2k7iCsl!OAA$tcb}yn-2-XrngniBSMpJHo*bvF1la}Y`y$j7InTNg`w#EQZWhn*h3a;~L=6^7H~%BI#ov{NQ_YN*dcb!aUr;iHt{~TA_q_JlDo;I<B{v%dNKemVp|f`|D00$p4t|erXR4IX>63^qLU(ewbF@5?a*H!#w<_HfflLt&(FERJ$OCN2M0i3+qyjR@c*X6+0r-wx056yttAN98KYpWeK@gmah+2m_VEUoUN9G^V4`;CB9O8l0!;6xNgZu3bQUmgM+vB+1O8<oV{skt6y^ib!i>ZQrU&UQ<5ulBEowUnxR0&m39dqyCN5juVZTgKi*}a##DTlag!2ZfD%TA$4>at_8Hzlrl3GXa4iZnTHim3sMisiy*wz+&(BD8DDW28~weEdQpkG_M==NBlUbS73?J14(QYG(odd-?T)SMU&R52&ZyCZ8qLrXS016a&xU4{CeJ8rI*0j`%{;On$!n8XvFM_iYn8c!P;k1G}upnIVA;&M;e?^Mv};ORxnQTIx5U#wET)-c7vYWo0;%Rdh$bLp0|qm7ShVft&t)@TECl{G|QGwBQ>N+u;=XJKJPwnduj%p;QE{pf$S~%IpPr50`0L`V^BUJm;Io*Rj=4Xd^w1?WP`2SnO&f=IFF&7(bBjbQPR{{XiGkm)dTbOom7csn2jPK7zYoeOB@u&n)TXx^Mmj7R2A>7ZuM^S7Gz~3|tZyYuSM$YK7KLvc}&E7zM4^aw=8Cv|MQ%bkQ=%VgDsMA(f^m-%I%?5F>$LWA_B+EX-G3)Fibwy$H>X`6vw{bBHiE>Fk_<Z3|^Cc_z1@s4g<9mh<0~Cz{)0kj|t})H#FtM6QMz;4!$*P9*Ci=U|0+&b*pB<4T7$z(jrpJ*#w+aE$n@Wy68;3%;ecqG(j<334LR1gQ(&iaAU#C?gh?bH-pd)#o@Wr()-|OZML#Z0gpO3<r}M<~VG>m>S~1QZpQ%>>E;NIu<*IMn27b&Yz+up;v|4@=a4FvdNn5)ug|*O1_tOkhdR`C=X{a?il4`z7+H>nO?lbZ-VXA2U0iWsyfaX&1C3a|8eX!5oMkyUjdiZ>VfZJE!za{mb}5=g(;C=1om+gk!)&@j5BkHUIrGxs8119NC-ETy}<WXUfIqFLFPBLpJW3wlnJDV9A%uNl3`8kslNqsh<HGb0AJ;FTi3ussoK6=&Z?~<dP>icdw!1|j#F5sL9<L?nK%^9Z`+Vf33Tu?Y=S)2Ifs42Oc0k+X+{iq)%GWSGh8o3uzq8nv4W{iwiixucIpJ22vBvcUIskzzp3$Z54th+3pmZZ1U12Vf2Lt2n&YqZsfv>x6}KkPhbR=5b6-R|Tub{v5915s5JCdytzm)J$OkGK<OU+W+vy9)TK9TW-@SckR5g?+@tR%@gd+qsnW+yC5&d|K@#!MGBsa79Jk#ROpmBV6&{MmFUMFt*&f`CkLHRp_Ve2q+*pbR!`K%fan=|eFk#IZJ#oC=|9f*lr%58wj%4D`h;wU7KDX?5;#uxH5<M{%X86+&`^Mr?Td)`^r2xX#jZ1;l!JQQAs^QcHFo5tZ;MdAmsU2NBwsl{uJdDwYo0e_tB0<Xc5V2GZ=ZDT&`33Oj|rf!v<%ll+aJ5GJZe9`Y&C2)!?Ah+>~I!(&qRxszjt-C1y2&M~Xz$n-Z_EkN~5>UjCwpx_zKqAXvbLjmr3~kJgA;R4SzNXZ2cw9TjEM~97b%#;Go1SpycX>3O&UDs?+v<XPNw2N96$j`L=%TIAO8m9k<^VS_rQ|!?Hvg``GGnsVOqope6&_LlntM^FfU1Ab^$YxqK0!8EH^*$pS&Lm6?>)}M(2La&fq?fVFLRTcj>j4C=(cYhIZ8v3C(JbVH2M#nY}u-h(QAT2xF5_whiSX?&cQ3ujrGR11;kGL2I%L!Yx)NcA=1?AeDz=l>NvZeT%&rBWq~qlII~$e22%sQ+0wvu>@v7c-P2AOUEnz|Ot)}(l*>y6+|+LAoci#azZT&Az)IFlR3=2myn<)<og|}_3wd4O;NV_$F`7lqp<;|Dg1h*k?KwR`oNoL|R2Yk4C(jviEIU;Q1B=<8{ngafa)29H{95Ql52sHGJM<Tzhi><;q4V`t>@2k0sMJJW++R2NCpds~Q?Fx&zZCA_?<5t=lki7+Z?Q8>l6WIq*v);FDP=!#lGwo5#oL~hR0U`D*Mx4Y8yn^c=uK=5054zB+u<Dz6b@odNJq_<yB1uAtG&3asXpdv+Xd!U`Vp;f*i5>4LKE*&xdAwq(14u84^Ruuldy{_K(4szYDe6g<E;#hoTg4v4T$crO8XUF^Ox(f_ypf+WDXjMHVnADgTO2-vF&ZNrG_&PwQJ1p+yu+SZ<NWSJ|dYqD_0Sl;Z5Q@^)&tml2!PZ`v>3R=t%IpZ)@Us(W*odSxTIDf%Z4wl0uMBI88o{ZBR`}JvG!A?E96-<jVK<CU`!do~GO)IvN9*Ew~k1i{8`k=&@R{-X4pPD_jlXPBLVZp&U2e<$d%r;e`KJ_7%2TvMP1S|MU#?h<}@Rko;InNa@V&RuRxu`Dm1ax7rr4Ld!+E`QzoSHk9m*H_Cm2-&PJ_c;J~)4Wv+^Oo9G=;BU2IX_&A9AFlr@rUi8Lj4cng^^d`y8d))ane*uF@-*<;BZI{-l^LM_V9d8oqwk5&sa8n6_#d>8!mG?7ZJf3Zu7vMWPT2?Bz9M^_Q?Wi|3ingePMK0ifF5F{Jd_yG;(y39<|bB?9xTMTj(Bpv@s^=<DLouJ*MC-c=!=1g^dbH&(n2}{*5XUS3bN9_O5O?Tu+6u;lqV8DBN0hW1dM*6#knrRbk13F-x`4JouirM-x9prAb{ozrM5de=>3_*d@Z<|Nc2xCj(7Ibfx1G!j<>>*vWw^qtO(yP_aoZDzd@#6gPRT95(GD}V)3c!3G@jT!>95LX0ZN>7zV>&Pmn6LRz?tJp#p5x{Ai|W9r26w(HqMg=JMrCVWT(_#8MXt96X?+lBcN6*g15#{EZHn&IV4uXDS2h`Fqp_CHvd2T=rNn&Al6r(LFKMkOKeC{MB>_JCeF(JE{-E1mdoAHrbK#n!AWpV>+QH;2pVzuer`eCAm(@<yOl32)!vhS4PO~nUQof+uysNOb7kt_qOQZEpS(R&Aw)99~@Hd&~9myO{+l_J;O%(UnNGkuX6*51L_P|gPlulO*E79<n2<GRvpy##}MbahJ;`BsGEp;YJA}Xb9bsKT&4bIJ%rz2Mlf`F2W_)H$QGiHW)|tAv1RNKXDjx!lC3U~YiNtmvq+)j_2S4l;`p~7sU%N&-l!CDmnc=k#A)!QzMZLO>ZW+B{Y(Ak&G!DZv!n7v87nmxwlI-GPo~2+l5-MvzBTl#O+g4}gJ+2@=mY&UbHkN@HD<B{qiqHf4Jo|_Sf^<8Ct*9b6b>xgn(qMXwG-S%i%yT#G_g{x71LAyBqvlfBOFSfz(RDB6tMNQJwRh@1Y1Eqvo;iJu}_TdaJg+En{B&CJ>_fX#Q@q!DxK-i107&Bthf8em^ct52g{?{oz;fo>F#iEFZma(E;TMKr2U)@pVMk~TF_yAde4r|4m~=~?l8S$(+(5bCv~{eahm*vye3a|k>(lNbLFVCC;Eo;1J}URq3vL{gdCx~<>uHoCY>zXc%XBzW*{f^ZOSDkMEt__;#&IX1H;R{vy~Bh?H|f6QhJ4MZkqI6s#dYRg5!qJ9|Fw+IdTD$W8jX~$<Urg{LW+q{mHuzzA8Ig{!iHyR~#|JzD4^lVXd;%o}BuO+ZD_Yy0oZ@9LL*3_bOfp+sNMtyzDktUAKd#sV}IH{wzkQkdm6D)~t9}K8QP%+~;7!;6!jNP?gZt7E{&`{z;rFZ;)y?x~1q99tUb)e(-N2lPRp2?>;NVm066U11TUeC9!;ES`RJD=Clt(2EyTG-il3%Be;wEj~;o*p}X|t0FkmdWeGBkt(W>U3o4LcKSr)-mAWCNMWCQ!8$l%3(t6lqjDf*Pj*-k1r6v8`2tAnPNGHeGy9MUijTAx8O8LFQeqc0AAUCRs2?Ju<5wqR@!J6z2V<G%MI_7+_"


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, 2 * D)
        self.fc2 = nn.Linear(2 * D, D)

    def forward(self, x):
        n = x.shape[1]
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(x.shape[0], n, H, D // H).transpose(1, 2)
        k = k.view(x.shape[0], n, H, D // H).transpose(1, 2)
        v = v.view(x.shape[0], n, H, D // H).transpose(1, 2)
        p = torch.arange(n, device=x.device)
        mask = (p[:, None] >= p[None, :]) & (p[:, None] - p[None, :] <= WINDOW)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(x.shape[0], n, D)
        x = x + self.proj(y)
        y = self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x + y


class Adder(nn.Module):
    def __init__(self):
        super().__init__()
        self.digit = nn.Embedding(10, D)
        self.role = nn.Embedding(3, D)
        self.blocks = nn.ModuleList([Block() for _ in range(N)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 10)

    def forward(self, tokens):
        roles = torch.arange(tokens.shape[1], device=tokens.device) % 3
        x = self.digit(tokens) + self.role(roles)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def _load(model):
    if not _WEIGHTS:
        return
    raw = zlib.decompress(base64.b85decode(_WEIGHTS.encode()))
    values = torch.frombuffer(bytearray(raw), dtype=torch.float16)
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(values[offset:offset + count].view(parameter.shape))
            offset += count


def build_model():
    model = Adder()
    _load(model)
    model.eval()
    return model, {
        "architecture": "2-layer local causal digit transformer",
        "parameters": sum(p.numel() for p in model.parameters()),
        "digits": 14,
        "training": "random full-length operand pairs"
    }


def _digits(value):
    result = []
    for _ in range(14):
        value, digit = divmod(value, 10)
        result.append(digit)
    return result


def _prefix(a, b):
    da, db = _digits(a), _digits(b)
    sequence = []
    for i in range(15):
        sequence.extend((da[i] if i < 14 else 0, db[i] if i < 14 else 0))
        if i < 14:
            sequence.append(0)
    return sequence


def _generate(model, a, b, width=5):
    da, db = _digits(a), _digits(b)
    beams = [([], 0.0)]
    device = next(model.parameters()).device
    for i in range(15):
        rows = []
        for output, score in beams:
            seq = []
            for j in range(i + 1):
                seq.extend((da[j] if j < 14 else 0, db[j] if j < 14 else 0))
                if j < i:
                    seq.append(output[j])
            rows.append(seq)
        x = torch.tensor(rows, dtype=torch.long, device=device)
        logp = model(x)[:, -1].log_softmax(-1)
        choices = []
        for row, (output, score) in enumerate(beams):
            vals, inds = logp[row].topk(width)
            for val, ind in zip(vals.tolist(), inds.tolist()):
                choices.append((output + [ind], score + val))
        choices.sort(key=lambda item: item[1], reverse=True)
        beams = choices[:width]
    return [output for output, _ in beams]


def _score(model, a, b, outputs):
    da, db = _digits(a), _digits(b)
    rows = []
    for output in outputs:
        seq = []
        for i in range(15):
            seq.extend((da[i] if i < 14 else 0, db[i] if i < 14 else 0, output[i]))
        rows.append(seq)
    device = next(model.parameters()).device
    x = torch.tensor(rows, dtype=torch.long, device=device)
    logits = model(x)[:, 1::3]
    targets = x[:, 2::3]
    return logits.log_softmax(-1).gather(2, targets.unsqueeze(-1)).squeeze(-1).sum(1)


def add(model, a: int, b: int) -> int:
    with torch.inference_mode():
        candidates = _generate(model, a, b) + _generate(model, b, a)
        unique = []
        seen = set()
        for candidate in candidates:
            key = tuple(candidate)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        scores = _score(model, a, b, unique) + _score(model, b, a, unique)
        output = unique[int(scores.argmax())]
    value = 0
    for digit in reversed(output):
        value = value * 10 + digit
    return value
