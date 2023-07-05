# crossmodal
Python code for the cross-modal retrieval system proposed by N. Rasiwasia, J.C. Pereira, E. Coviello, G. Doyle, G.R.G. Lanckriet, R. Levy, N. Vasconcelos in the ACM MultiMedia '10 paper 
["A New Approach to Cross-Modal Multimedia Retrieval"](http://acsweb.ucsd.edu/~ecoviell/pubs/ANewApproachtoCross-ModalMultimediaRetrieval.pdf).


Dependencies
------------

numpy
sklean 0.16
scipy.io (to read .mat files)

Todo
----

* remove dependency on .mat format
* refactor code to run multiple experiments (CM, SM, SCM)

Citing
------

When using the code or the dataset, please cite:


@inproceedings{rasiwasia2010new,

  title={A new approach to cross-modal multimedia retrieval},
  
  author={Rasiwasia, Nikhil and Costa Pereira, Jose and Coviello, Emanuele and Doyle, Gabriel and Lanckriet, Gert RG and Levy, Roger and Vasconcelos, Nuno},
  
  booktitle={Proceedings of the international conference on Multimedia},
  
  pages={251--260},
  
  year={2010},
  
  organization={ACM}
  
}

@article{costa2014role,

  title={On the role of correlation and abstraction in cross-modal multimedia retrieval},
  
  author={Costa Pereira, Jose and Coviello, Emanuele and Doyle, Gabriel and Rasiwasia, Nikhil and Lanckriet, Gert RG and Levy, Roger and Vasconcelos, Nuno},
  
  journal={Pattern Analysis and Machine Intelligence, IEEE Transactions on},
  
  volume={36},
  
  number={3},
  
  pages={521--535},
  
  year={2014},
  
  publisher={IEEE}
  
}



