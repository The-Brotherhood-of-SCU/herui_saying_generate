# AI HeRui

## Dependency
pytorch

## Attention
if you have not installed `cuda`,please set the varible `device` to "cpu" in `herui_saying_generate.py`

## Use
if you directly execute `herui_saying_generate.py`,it will train the model,and then you can call `generate` function

eg.
```py
generate("川大")# it will generate saying starting with the parameter
```
