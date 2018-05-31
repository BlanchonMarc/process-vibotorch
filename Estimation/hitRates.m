function [hitRate , falseAlarm, hitCount, missCount] = hitRates(testMap,gtMap)
% code from Margolin et al. CVPR 2013

neg_gtMap = ~gtMap;
neg_testMap = ~testMap;

hitCount = sum(sum(testMap.*gtMap));
trueAvoidCount = sum(sum(neg_testMap.*neg_gtMap));
missCount = sum(sum(testMap.*neg_gtMap));
falseAvoidCount = sum(sum(neg_testMap.*gtMap));

falseAlarm = 1 - trueAvoidCount / (eps+trueAvoidCount + missCount);

hitRate = hitCount / (eps+ hitCount + falseAvoidCount);
