Extension

- Trying to solve same method , different areas to attack.
- Trying to Solve the limitations section.
- Improving the performance metric.
- Algorithm vs Metric.

- check into intervention time methods
  - they dont leverage that SAEs are not interpretable
  - only checking what the SAEs we want and what we dont want
  - Is it possible to check "<harmful content>" which SAE to do the unlearning
  and doing evals on that

- Mech Interp (White Box Analysis)
  - an analysis type of thing -> take an existing problem or method and try to study it better
   -> what model circuits change between the normal and unlearned version
   -> does it breaks the conenctions or representations still there or not ?
  - Taking two parts and comparing them (preunlearning and postunlearning -> models, activations and weights , how do they differ , what are the feature changes)
  - question them why that is the case ?

- Non SAE methods
  - What SAE features correspond to the thing that we want to unlearn,
  are what directions the dense versions corresponds and how do those differ ?
  - difference tell what is the need to do SAE.
  - while learning dense direction, can have a dense direction to detech sparsity.

  
