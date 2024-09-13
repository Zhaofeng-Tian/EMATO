

# Energy-Model-Aware Trajectory Optimization for Autonomous Driving

## Features
<center>
  <img src="/imgs/Fig1.png" alt="EMATO Framework in a Frenet System" width="500" />
</center>
  
- Differentiable energy model 
- Online nonlinear programming (NLP) optimization
- Integration of traffic and road slope predictions
- Extensive validation through case studies and quantitative analysis
## How to Use

Install [CasADi](https://www.openai.com "Visit OpenAI's website") in your Python environment.

For the Frenet environment, use:

```python emato/emato/sim/frenet.py```

For the ACC test:

```python emato/emato/sim/acc.py```

