(* Export the 95-600 GeV binary DY data to JSON for the Python pipeline. *)
Get["/Users/user/Downloads/dy_eftvalid_data.m"];
coords = d6coeff; nC = Length[coords];
premodel = AMDYbinnedfinal /. Thread[d8coeff -> 0];
smodel = Table[premodel[[i, 3]], {i, Length[premodel]}];
nBins = Length[smodel];
NSM = SMvalarray[[All, 3]];
A = N@Table[
  Coefficient[smodel[[b]], coords[[i]]] /. Thread[coords -> ConstantArray[0., nC]],
  {b, nBins}, {i, nC}];
Q = N@Table[
  Module[{tmp},
    tmp = Coefficient[Coefficient[smodel[[b]], coords[[i]]], coords[[j]]] /.
      Thread[coords -> ConstantArray[0., nC]];
    If[i == j, 2 tmp, tmp]],
  {b, nBins}, {i, nC}, {j, nC}];
Export["/Users/user/Downloads/dy_real_95_600.json",
  <|"names" -> ToString /@ coords,
    "binlo" -> SMvalarray[[All, 1]], "binhi" -> SMvalarray[[All, 2]],
    "NSM" -> NSM, "A" -> A, "H" -> Q|>, "JSON"];
Print["exported ", nBins, " bins x ", nC, " coeffs -> dy_real_95_600.json"];
