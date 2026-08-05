(* Export the dim-8 linear sensitivities of the 95-600 GeV DY data:      *)
(* A8[b, j] = d mu_b / d c8_j at c = 0 (dim-8 interference, O(1/Lambda^4) *)
(* like dim6^2). For the C8 nuisance gate.                                *)
Get["/Users/user/Downloads/dy_eftvalid_data.m"];
smodelFull = Table[AMDYbinnedfinal[[i, 3]], {i, Length[AMDYbinnedfinal]}];
nBins = Length[smodelFull];
zeroAll = Join[Thread[d6coeff -> 0], Thread[d8coeff -> 0]];
A8 = N@Table[
  Coefficient[smodelFull[[b]], d8coeff[[j]]] /. zeroAll,
  {b, nBins}, {j, Length[d8coeff]}];
(* also the d6-d8 cross terms exist at O(1/Lambda^6): ignored (higher order) *)
Export["/Users/user/Downloads/dy_dim8.json",
  <|"names8" -> ToString /@ d8coeff, "A8" -> A8|>, "JSON"];
Print["exported A8: ", Dimensions[A8], " -> dy_dim8.json"];
Print["column norms (first 8): ",
  Table[Norm[A8[[All, j]]], {j, 8}]];
