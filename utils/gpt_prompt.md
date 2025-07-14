There is a following file structure:
```fs
/path/to/generation/
-{prompt-name}/ # name of prompt that was used during generation
--output/
---orig.png # generation without proposed guidance method with whole 28 steps
--output_{N}/ # generation with N steps
---smal.png # without proposed guidance method (equal to guidance scale == 2)
---skip_steps_{M}/
----skip_{coef}.png # with guidance scale of the proposed method equal to 'coef'
```

I need you to create a grid of each generation for each N:
- in each row there should be images with the same 'M' value
- to the left of them there should be orig.png with size equal to all other images size - but alone in the middle of a column, marked with a red rectangle
- smal.png should be pasted into the line of coefs as a coef == 2
- between all the images there should be no white space - like just tensor concatenation, but with images
- the coef values should be written under the grid in the middle of a corresponding column width - within the same N the 'coef' values for each 'M' are the same
- the 'M' values should be written to the left of the grid in the middle of a corresponding row height
- save the grid images as ```/path/to/generation/output_{N}/grid.png```