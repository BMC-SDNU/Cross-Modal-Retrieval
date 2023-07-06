function [logPot] = LLM_logPot(param,y,w1,w2,edges)

nNodes = size(w1,1);
nStates = size(w1,2)+1;
nEdges = size(edges,1);

logPot = 0;
for n = 1:nNodes
    if y(n) < nStates
        logPot = logPot + w1(n,y(n));
    end
end
for e = 1:nEdges
   n1 = edges(e,1);
   n2 = edges(e,2);
   switch param
       case 'C'
           if y(n1)==1 && y(n2) == 1
               logPot = logPot + w2(e);
           end
       case 'I'
           if y(n1)==y(n2)
               logPot = logPot + w2(e);
           end
       case 'P'
           if y(n1) == y(n2)
               logPot = logPot + w2(y(n1),e);
           end
       case 'S'
           if mod(y(n1)+y(n2),2)
               logPot = logPot + w2(e);
           end
       case 'F'
           logPot = logPot + w2(y(n1),y(n2),e);
   end     
end
    