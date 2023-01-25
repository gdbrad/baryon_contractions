# baryon_contractions
Original author: Amy Nicholson

To run tests against known results using ```baryon_contractions_gb.py``` and ```test_baryons_gb.py```:
```
chmod +x run_test.sh
./run_test.sh
```
Included directories:
- lalibe_src: 
  corresponding proton/baryon contraction code for lalibe 
- known_results:
   h5 files needed to run cross-check 
- notes_chroma:
   A few .tex/pdf notes from the chroma src code dealing with various contraction subroutines

Included files:
- proton_FH_contractions.tex:
  original notes by Andre, David in which I have appended info for delta baryons 
- gamma.py: 
  gamma matrix construction
  
## Unresolved questions:

- Construction of src/sink weight matrices:
  - how does this differ for each baryon? 
  - Is there a direct mapping from lines 67-756 in ```lalibe_src/baryon_contractions_func_w.cc``` to the src/snk spin matrices constructed for each isospin
  in ```baryon_contractions_gb.py```?

- 4/3 coeff for xi baryons 





TODO for Lalibe: 
Currently no FH propagators
