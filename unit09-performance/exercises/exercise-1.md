## Exercise 1: Evaluating Mixed-Precision Performance

I recommend trying out the different mixed-precision choices on your GPU (if you have one): 

- regular float32 (`"32-true"`)

- regular float16 (`"16-true"`)

- regular float64 (`"64-true"`)

- mixed precision with float16 (`"16-mixed"`)

- mixed-precision with bfloat16 (`"bf16-mixed"`)

For this, you can modify the `precision` argument of the `Trainer` in the [DistillBert model from lecture 9.1here](https://github.com/Lightning-AI/dl-fundamentals/tree/main/unit09-performance/9.1-mixed-precision/part2-distilbert-example).

Please also fell free to [share your results in the discussion here](https://github.com/Lightning-AI/dl-fundamentals/discussions/52) -- I'd be interested to hear what you find on various GPU architectures!