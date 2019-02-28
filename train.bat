ipython nbconvert --to python Latium.ipynb
python Latium.py %*
:repeat
python Latium.py --load %*
goto repeat