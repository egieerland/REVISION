flowchart LR
    %% Input data from sensors
    Sensors([Sensor Data]) --> ProcessData{Processing}
    
    %% Different processing approaches
    ProcessData -->|Rule-Based| Rules[IF-THEN Rules]
    ProcessData -->|Bayesian| Bayes[P(H|E) ∝ P(E|H)P(H)]
    ProcessData -->|Dempster-Shafer| DS[Mass Functions + Combination]
    ProcessData -->|ML Models| ML[Feature Vector → Model]
    ProcessData -->|Sequence| Seq[Temporal Features → Model]
    ProcessData -->|Hybrid| Hybrid[ML + Safety Rules]
    
    %% Outputs
    Rules --> Out[Decision Output]
    Bayes --> Out
    DS --> Out
    ML --> Out
    Seq --> Out
    Hybrid --> Out
    
    %% Special handling
    Bayes -.-> Momentum[Temporal Momentum]
    DS -.-> Conflict[Conflict K]
    ML & Seq -.-> Fallback[Safety Fallback]
    Hybrid -.-> Override[Safety Override]
    
    %% Uncertainty representation
    Bayes --> PostProb[Posteriors]
    DS --> BeliefPlaus[Belief/Plausibility]
    ML --> Confidence[Confidence Scores]
    Seq --> TempConf[Temporal Confidence]
    Hybrid --> MultiConf[Multi-level Confidence]
    
    %% Style
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef process fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef special fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    
    class Sensors input
    class Rules,Bayes,DS,ML,Seq,Hybrid process
    class Out output
    class Momentum,Conflict,Fallback,Override special
    class PostProb,BeliefPlaus,Confidence,TempConf,MultiConf special