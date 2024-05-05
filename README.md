# Xylophone

I would like to build a xylophone. It is more involved than I expected.

## References

### Supermediocre
Rich Wickstrom has built a beautiful 3.5-octave 44-bar xylophone.
The tone bars were made of rosewood, which is the standard for professional-quality tone bars.
He used MATLAB to optimise the shape.

http://supermediocre.org/index.php/rich/richs-projects/xylophone-project/

### Kate Salesin
Kate Salesin has built a beautiful 2.5-octave 32-key xylophone.
The tone bars were made of padauk, which is generally considered a good wood for tone.
She used MATLAB to optimise the shape.

https://medium.com/@kas493/building-a-xylophone-part-1-xylo-troduction-a9e914ddaa13

### Mingming
Zhao Mingming wrote a 2011 masters thesis at Curtin university, under supervision of Rodney Entwistle.
This work describes using a CNC machine to predict tuning curves for wooden tone bars, including a fine-tuning process that accounts for the natural variance in the wood.
For each bar, the fine-tuning process requires a few uncalibrated cuts of successive depth; after each cut, the bar is manually struck and its frequencies analysed.
After five uncalibrated cuts, the operator performs linear regression to account for the difference between the predicted and observed frequencies.
Then, this regression is used to calibrate the final cut.

This is the only work that I am aware of that deals with the problem of *wood*.

https://espace.curtin.edu.au/bitstream/handle/20.500.11937/68/179150_Zhao2011.pdf?sequence=2&isAllowed=y

### Entwistle 
Timoshenko beam theory for aluminium bars.

https://www.acoustics.asn.au/conference_proceedings/ICSV14/papers/p248.pdf