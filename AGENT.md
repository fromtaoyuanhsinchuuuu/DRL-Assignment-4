# Agent Guidelines for DRL-Assignment-4

## Build/Test Commands

### Q1 (Pendulum-v1)
- Train: `python Q1/train.py`
- Evaluate: `python Q1/eval.py --episodes 100`
- Record demo: `python Q1/eval.py --record_demo`

### Q2 (DMC environment)
- Train: `python Q2/train.py`
- Evaluate: `python Q2/eval.py`

### Q3 (SAC with humanoid-walk)
- Train: `python Q3/train.py`
- Resume training: `python Q3/train.py --resume --checkpoint final`
- Evaluate: `python Q3/eval.py`

## Code Style Guidelines

- **Imports**: Group imports by stdlib, external packages, then local modules
- **Formatting**: Use 4-space indentation, no tabs
- **Naming**: Use snake_case for variables and functions, CamelCase for classes
- **Documentation**: Use docstrings for functions with parameters and return values
- **Error handling**: Use try/except blocks with specific error types
- **Type hints**: Not consistently used but encouraged for new code
- **Line length**: Keep lines under 100 characters
- **Model checkpoints**: Save models to designated model directories