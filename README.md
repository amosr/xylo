# Xylophone

I would like to build a xylophone. It is more involved than I expected.

## Building

This requires Python 3 with jax and jupyter.

```
# Create a local environment
python3 -m venv .venv
# Enable the virtual environment
source .venv/bin/activate
# Install dependencies
pip3 install jupyter matplotlib
pip3 install scipy numpy jax optimistix
pip3 install evosax jax_cosmo
```


## References

### Supermediocre
Rich Wickstrom has built a beautiful 3.5-octave 44-bar xylophone.
The tone bars were made of rosewood, which is the standard for professional-quality tone bars.

Rich used a MATLAB implementation of Timoshenko beam theory to optimise the shape.
I based the code here closely off his implementation, which was in turn based on Mingming's masters thesis.

However, I wasn't able to get the MATLAB code to run, so I re-implemented parts of it in Python.

http://supermediocre.org/index.php/rich/richs-projects/xylophone-project/

### Mingming thesis
Zhao Mingming wrote a 2011 masters thesis at Curtin university, under supervision of Rodney Entwistle.
This work describes using a CNC machine to predict tuning curves for wooden tone bars, including a fine-tuning process that accounts for the natural variance in the wood.
For each bar, the fine-tuning process requires a few uncalibrated cuts of successive depth; after each cut, the bar is manually struck and its frequencies analysed.
After five uncalibrated cuts, the operator performs linear regression to account for the difference between the predicted and observed frequencies.
Then, this regression is used to calibrate the final cut.

This is the only work that I am aware of that deals with the problem of *wood*.

https://espace.curtin.edu.au/bitstream/handle/20.500.11937/68/179150_Zhao2011.pdf?sequence=2&isAllowed=y

### Kate Salesin
Kate Salesin has built a beautiful 2.5-octave 32-key xylophone.
The tone bars were made of padauk, which is generally considered a good wood for tone.

Kate used a MATLAB implementation of a 3D finite elements analysis to optimise the shape.
This approach is more accurate than the Timoshenko beam theory mentioned above and in Mingming's thesis.
However, it doesn't take into account the non-uniform structure of the wood, while Mingming's fine-tuning process does.
My hypothesis is that the extra accuracy gained from using a 3D analysis will be dwarfed by the anisotropy of the wood, so I'm going to use Mingming's fine-tuning process instead.

(I also wonder whether it's possible to adapt Mingming's fine-tuning to 3D analysis.)

https://medium.com/@kas493/building-a-xylophone-part-1-xylo-troduction-a9e914ddaa13

### Entwistle 
Timoshenko beam theory for aluminium bars.

https://www.acoustics.asn.au/conference_proceedings/ICSV14/papers/p248.pdf

### Beaton and Scavone

Computational design and simulation of idiophone bars, Douglas Beaton, PhD thesis.

This thesis optimises 3d bar shapes to find the best *torsional* modes in addition to the usual lateral modes.
Timoshenko beam theory ignores the torsional modes.
If left untuned, these torsional modes can produce undesirable beating and interfere with the fundamental.

The thesis experimentally validates the designs on aluminium bars, but not on wood bars.
It's not clear to me how closely the torsional modes match the optimised bar for wooden bars.
For now, I will follow Mingming's thesis, which explicitly accounts for differences in woods.

https://pubs.aip.org/asa/jasa/article-abstract/149/6/3758/945967/Three-dimensional-tuning-of-idiophone-bar-modes?redirectedFrom=fulltext
