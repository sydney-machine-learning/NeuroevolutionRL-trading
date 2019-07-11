
#!/bin/sh 
echo Running all problems.

for prob in  'iris' 'cancer' 'ions' 'penDigit'
	do
	python evo_pt_mcmc.py --problem $prob 
done