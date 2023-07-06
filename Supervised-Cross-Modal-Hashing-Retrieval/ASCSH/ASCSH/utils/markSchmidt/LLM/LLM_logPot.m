function [logPot] = LLM_logPot(param,y,w1,w2,w3,w4,w5,w6,w7,edges2,edges3,edges4,edges5,edges6,edges7)

nNodes = size(w1,1);
nStates = size(w1,2)+1;
nEdges2 = size(edges2,1);
nEdges3 = size(edges3,1);
nEdges4 = size(edges4,1);
nEdges5 = size(edges5,1);
nEdges6 = size(edges6,1);
nEdges7 = size(edges7,1);

logPot = 0;
for n = 1:nNodes
    if y(n) < nStates
        logPot = logPot + w1(n,y(n));
    end
end
for e = 1:nEdges2
   n1 = edges2(e,1);
   n2 = edges2(e,2);
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
for e = 1:nEdges3
   n1 = edges3(e,1);
   n2 = edges3(e,2);
   n3 = edges3(e,3);
   switch param
       case 'C'
           if y(n1)==1 && y(n2)==1 && y(n3)==1
               logPot = logPot + w3(e);
           end
       case 'I'
           if y(n1)==y(n2) && y(n2)==y(n3)
               logPot = logPot + w3(e);
           end
       case 'P'
           if y(n1)==y(n2) && y(n2)==y(n3)
               logPot = logPot + w3(y(n1),e);
           end
       case 'S'
           if mod(y(n1)+y(n2)+y(n3),2)
               logPot = logPot + w3(e);
           end
       case 'F'
           logPot = logPot + w3(y(n1),y(n2),y(n3),e);
   end
end
for e = 1:nEdges4
   n1 = edges4(e,1);
   n2 = edges4(e,2);
   n3 = edges4(e,3);
   n4 = edges4(e,4);
   switch param
       case 'C'
           if y(n1)==1 && y(n2)==1 && y(n3)==1 && y(n4)==1
               logPot = logPot + w4(e);
           end
       case 'I'
           if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4)
               logPot = logPot + w4(e);
           end
       case 'P'
           if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4)
               logPot = logPot + w4(y(n1),e);
           end
       case 'S'
           if mod(y(n1)+y(n2)+y(n3)+y(n4),2)
               logPot = logPot + w4(e);
           end
       case 'F'
           logPot = logPot + w4(y(n1),y(n2),y(n3),y(n4),e);
   end
end
for e = 1:nEdges5
   n1 = edges5(e,1);
   n2 = edges5(e,2);
   n3 = edges5(e,3);
   n4 = edges5(e,4);
   n5 = edges5(e,5);
   switch param
       case 'C'
           if y(n1)==1 && y(n2)==1 && y(n3)==1 && y(n4)==1 && y(n5)==1
               logPot = logPot + w5(e);
           end
       case 'I'
           if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5)
               logPot = logPot + w5(e);
           end
       case 'P'
           if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5)
               logPot = logPot + w5(y(n1),e);
           end
       case 'S'
           if mod(y(n1)+y(n2)+y(n3)+y(n4)+y(n5),2)
               logPot = logPot + w5(e);
           end
       case 'F'
           logPot = logPot + w5(y(n1),y(n2),y(n3),y(n4),y(n5),e);
   end
end
for e = 1:nEdges6
   n1 = edges6(e,1);
   n2 = edges6(e,2);
   n3 = edges6(e,3);
   n4 = edges6(e,4);
   n5 = edges6(e,5);
   n6 = edges6(e,6);
   switch param
       case 'C'
           if y(n1)==1 && y(n2)==1 && y(n3)==1 && y(n4)==1 && y(n5)==1 && y(n6)==1
               logPot = logPot + w6(e);
           end
       case 'I'
           if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5) && y(n5)==y(n6)
               logPot = logPot + w6(e);
           end
       case 'P'
           if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5) && y(n5)==y(n6)
               logPot = logPot + w6(y(n1),e);
           end
       case 'S'
           if mod(y(n1)+y(n2)+y(n3)+y(n4)+y(n5)+y(n6),2)
               logPot = logPot + w6(e);
           end
       case 'F'
           logPot = logPot + w6(y(n1),y(n2),y(n3),y(n4),y(n5),y(n6),e);
   end
end
for e = 1:nEdges7
   n1 = edges7(e,1);
   n2 = edges7(e,2);
   n3 = edges7(e,3);
   n4 = edges7(e,4);
   n5 = edges7(e,5);
   n6 = edges7(e,6);
   n7 = edges7(e,7);
   switch param
       case 'C'
           if y(n1)==1 && y(n2)==1 && y(n3)==1 && y(n4)==1 && y(n5)==1 && y(n6)==1 && y(n7) == 1
               logPot = logPot + w7(e);
           end
       case 'I'
           if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5) && y(n5)==y(n6) && y(n6)==y(n7)
               logPot = logPot + w7(e);
           end
       case 'P'
           if y(n1)==y(n2) && y(n2)==y(n3) && y(n3)==y(n4) && y(n4)==y(n5) && y(n5)==y(n6) && y(n6)==y(n7)
               logPot = logPot + w7(y(n1),e);
           end
       case 'S'
           if mod(y(n1)+y(n2)+y(n3)+y(n4)+y(n5)+y(n6)+y(n7),2)
               logPot = logPot + w7(e);
           end
       case 'F'
           logPot = logPot + w7(y(n1),y(n2),y(n3),y(n4),y(n5),y(n6),y(n7),e);
   end
end
    