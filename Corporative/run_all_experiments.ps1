# Run all experiments sequentially
# Usage: .\run_all_experiments.ps1 [-Episodes <int>] [-EvalEpisodes <int>]

param (
    [int]$Episodes = 20000,
    [int]$EvalEpisodes = 100
)

$algos = @(
    @{ Name = "DQLearning"; TrainScript = "train_dqn.py"; EvalScript = "eval_dqn.py" },
    @{ Name = "DDQLearning"; TrainScript = "train_ddqn.py"; EvalScript = "eval_ddqn.py" },
    @{ Name = "Monte_Carlo_on_policy"; TrainScript = "train_monte_carlo.py"; EvalScript = "eval_monte_carlo.py" },
    @{ Name = "SARSA_On_Policy"; TrainScript = "train_sarsa.py"; EvalScript = "eval_sarsa.py" },
    @{ Name = "QLearning"; TrainScript = "train_qlearning.py"; EvalScript = "eval_qlearning.py" }
)

$baseDir = Get-Location

foreach ($algo in $algos) {
    $algoDir = Join-Path "Algos" $algo.Name

    if (Test-Path $algoDir) {
        Write-Host "`n========================================================" -ForegroundColor Cyan
        Write-Host "Processing $($algo.Name)..." -ForegroundColor Cyan
        Write-Host "========================================================" -ForegroundColor Cyan

        Push-Location $algoDir

        # 1. Run Training
        Write-Host "Starting training for $($Episodes) episodes..." -ForegroundColor Yellow
        $trainCmd = "python $($algo.TrainScript) --episodes $Episodes"
        Invoke-Expression $trainCmd

        if ($LASTEXITCODE -eq 0) {
            # 2. Find the latest experiment folder
            if (Test-Path "models") {
                $latestExp = Get-ChildItem "models" | Where-Object { $_.PSIsContainer } | Sort-Object CreationTime -Descending | Select-Object -First 1

                if ($latestExp) {
                    $bestModelPath = Join-Path "models" $latestExp.Name "best_model.pkl"
                    $finalModelPath = Join-Path "models" $latestExp.Name "final_model.pkl"

                    $modelToEval = $null

                    if (Test-Path $bestModelPath) {
                        $modelToEval = $bestModelPath
                        Write-Host "Found best model: $modelToEval" -ForegroundColor Green
                    } elseif (Test-Path $finalModelPath) {
                        $modelToEval = $finalModelPath
                        Write-Host "Best model not found, using final model: $modelToEval" -ForegroundColor Yellow
                    }

                    # 3. Run Evaluation
                    if ($modelToEval) {
                        Write-Host "Starting evaluation for $($EvalEpisodes) episodes..." -ForegroundColor Yellow
                        $evalCmd = "python $($algo.EvalScript) --model-path $modelToEval --episodes $EvalEpisodes"
                        Invoke-Expression $evalCmd
                    } else {
                        Write-Host "Error: No model file found in $($latestExp.FullName)" -ForegroundColor Red
                    }
                } else {
                    Write-Host "Error: No experiment folder found in models/" -ForegroundColor Red
                }
            } else {
                Write-Host "Error: 'models' directory not found in $($algoDir)" -ForegroundColor Red
            }
        } else {
            Write-Host "Training failed for $($algo.Name)" -ForegroundColor Red
        }

        Pop-Location
    } else {
        Write-Host "Directory not found: $algoDir" -ForegroundColor Red
    }
}

Write-Host "`nAll experiments completed." -ForegroundColor Cyan
