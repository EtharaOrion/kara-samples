# Task List

1. ✅ Inspect baseline files and runtime/GPU environment
Workspace empty; H100 80GB available; torch 2.6 under /usr/local/bin/python3.
2. ✅ Create complete inference submission at graded path
Created complete inference implementation early; final contains trained source-literal weights.
3. ✅ Train robust transformer and staged FFN compression
Width-4 teacher, width-3 and width-2 recovery complete. Final exact on 500k uniform + 500k structured.
4. ✅ Apply exact gauge/symmetry parameter reductions
Applied exact positional/K/V/LayerNorm/logit invariances. Final model has 2,343 parameters.
5. ✅ Validate exported submission thoroughly
Fresh import passed; 500k random, 500k structured, 38,809 edge, 2,000 API all exact; attention and output ablations each changed 40/40.
