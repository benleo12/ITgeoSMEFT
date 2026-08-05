(* Extract Adam's SMEFT_LEP.nb: lep1inputs (11 observables), chisqLEP,     *)
(* coefficient lists; apply his own reduction pipeline (Series to O(x^2),  *)
(* dim8 -> 0, bb-merge, cHB=cHW=0); export per-observable A/B/Q and the    *)
(* chi^2 Hessian at the best fit to JSON for the Python pipeline.          *)

nbfile = "/Users/user/Library/CloudStorage/Dropbox/ITGeo (1)/SMEFT_LEP.nb";
nb = Get[nbfile];
cells = Cases[nb, Cell[BoxData[b_], "Input", ___] :> b, Infinity];
Print["input cells: ", Length[cells]];

(* evaluate the assignment cells we need: 1 (lep1inputs), 2 (chisqLEP),   *)
(* 5 (dim8coefftozero), 7 (d6coeff)                                        *)
Do[ToExpression[cells[[k]], StandardForm], {k, {1, 2, 5, 7}}];
Print["lep1inputs: ", Length[lep1inputs], " observables"];
Print["d6coeff: ", d6coeff];
Print["dim8 list length: ", Length[dim8coefftozero]];

(* Adam's own reduction (cell 9 / 17 pipeline) *)
reduce[expr_] := Normal[Simplify[
  Series[expr, {x, 0, 2}] /. Thread[dim8coefftozero -> 0] /.
    c36Hψqbb -> c36Hψq /. c16Hψqbb -> c16Hψq /. cHd6bb -> cHd6 /.
    cHB -> 0 /. cHW -> 0]];

lep16 = reduce[lep1inputs];
chilep = reduce[chisqLEP];

(* residual-symbol audit: anything left besides d6coeff and x? *)
syms = Union[Cases[{lep16, chilep}, s_Symbol /;
   ! MemberQ[Attributes[s], Protected], Infinity]];
extra = Complement[syms, Join[d6coeff, {x}]];
Print["residual symbols beyond d6coeff+x: ", extra];
(* zero any stragglers (dim-8 typos in the notebook list) *)
If[Length[extra] > 0,
  lep16 = lep16 /. Thread[extra -> 0];
  chilep = chilep /. Thread[extra -> 0]];

x0 = (0.246/5)^2;
nO = Length[lep16]; nC = Length[d6coeff];
zeroSub = Thread[d6coeff -> ConstantArray[0, nC]];

(* per-observable structure at x = x0:                                     *)
(* mu_o = S_o + A[o,i] c_i + Q[o,i,j] c_i c_j  (after x->x0)               *)
lepx = lep16 /. x -> x0;
SM = N[lepx /. zeroSub];
Aarr = N@Table[Coefficient[lepx[[o]], d6coeff[[i]]] /. zeroSub,
   {o, nO}, {i, nC}];
Qarr = N@Table[Module[{t},
    t = Coefficient[Coefficient[lepx[[o]], d6coeff[[i]]], d6coeff[[j]]] /.
      zeroSub;
    If[i == j, 2 t, t]], {o, nO}, {i, nC}, {j, nC}];

(* chi^2: best fit + Hessian at best fit (with |c| < 4 pi box, x -> x0)   *)
chix = chilep /. x -> x0;
vars = d6coeff;
themin = Quiet[FindMinimum[
   {chix, Thread[Abs[vars] < 4 Pi]},
   Table[{vars[[i]], 0}, {i, nC}]]];
best = vars /. themin[[2]];
Print["chi2 at SM: ", N[chix /. zeroSub], "   at best fit: ", themin[[1]]];
Print["best fit: ", Thread[d6coeff -> NumberForm[#, 3] & /@ best]];
hess = N@Table[(1/2) D[chix, vars[[i]], vars[[j]]] /. themin[[2]],
   {i, nC}, {j, nC}];
(* Hessian and gradient at the SM point c=0: the clean Fisher reference   *)
(* (best fit above pins 3 coefficients at the 4pi box boundary, so the    *)
(* Hessian there is not a stationary-point Fisher).                        *)
hess0 = N@Table[(1/2) D[chix, vars[[i]], vars[[j]]] /. zeroSub,
   {i, nC}, {j, nC}];
grad0 = N@Table[D[chix, vars[[i]]] /. zeroSub, {i, nC}];

Export["/Users/user/Downloads/lep_real.json",
  <|"names" -> ToString /@ d6coeff,
    "x0" -> x0,
    "SM" -> SM, "A" -> Aarr, "Q" -> Qarr,
    "chi2SM" -> N[chix /. zeroSub], "chi2min" -> themin[[1]],
    "bestfit" -> best, "hessian" -> hess,
    "hessian0" -> hess0, "grad0" -> grad0|>, "JSON"];
Print["exported -> lep_real.json  (", nO, " observables x ", nC, " coeffs)"];
