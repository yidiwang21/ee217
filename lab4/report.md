## Lab4 Report
1. Execution time without streams: 2.40697 ms; Execution time with streams: 1.22699 ms

2. Yes.
![alt text](https://github.com/yidiwang21/ee217/blob/master/lab4/screenshot/Screen%20Shot%202019-03-04%20at%2010.42.10%20AM.png)

3. Yes. The first kernel is now running while the second one is doing MemCpy(HtoD). When the second kernel is running, the third one is doing MemCpy(HtoD).

4. Yes. More memory copy operations will overlap.]
![alt text](https://github.com/yidiwang21/ee217/blob/master/lab4/screenshot/Screen%20Shot%202019-03-04%20at%2010.43.47%20AM.png)
