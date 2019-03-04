1. 
Execution time without streams: 2.40697 ms
Execution time with streams:    1.22699 ms

2. Yes.

3. Yes. The first kernel is now running while the second one is doing MemCpy(HtoD). When the second kernel is running, the third one is doing MemCpy(HtoD).

4. Yes. More memory copy operations will overlap.
