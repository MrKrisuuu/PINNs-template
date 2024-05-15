def get_initial_conditions(problem):
    if problem == "SIR":
        S = 0.9
        I = 0.1
        R = 0.0
        params = (2, 1)
        return (S, I, R, params)
    else:
        raise Exception("Wrong problem!")