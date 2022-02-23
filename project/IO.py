

def getPrintProbString(step, outcome, prob_add_zero, prob_limit, flipped_coin):
    return (
        f'Step {step}\n' 
        f'-------------\n'  
        f'Current outcome is |{outcome}>\n'  
        f'Probability of adding 0 is {prob_add_zero:.3f}\n' 
        f'Probability of adding 1 is {(prob_limit-prob_add_zero):.3f}\n'
        f'p({outcome}0) + p({outcome}1) = {prob_limit:.3f}\n'
        f'Random number generated: {flipped_coin}'
    )