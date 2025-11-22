# Run all experiments for competitive algorithms sequentially
# Usage: .\run_all_experiments.ps1 [-Episodes <int>] [-TestEpisodes <int>] [-SaveInterval <int>]

param (
    [int]$Episodes = 3000,
    [int]$TestEpisodes = 100,
    [int]$SaveInterval = 20.
)

$algos = @(
    @{ Name = "deep_nash_q"; TrainScript = "train_deep.py" },
    @{ Name = "nash_q"; TrainScript = "train.py" }
)

$baseDir = Get-Location

foreach ($algo in $algos) {
    $algoDir = Join-Path "Algos" $algo.Name

    if (Test-Path $algoDir) {
        Write-Host "`n========================================================" -ForegroundColor Cyan
        Write-Host "Processing $($algo.Name)..." -ForegroundColor Cyan
        Write-Host "========================================================" -ForegroundColor Cyan

        Push-Location $algoDir

        # Create a timestamped experiment directory for this algorithm
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $expDir = Join-Path $baseDir (Join-Path "experiments" "$($algo.Name)_$timestamp")
        New-Item -ItemType Directory -Force -Path $expDir | Out-Null

        # 1. Run Training
        Write-Host "Starting training for $($Episodes) episodes..." -ForegroundColor Yellow
        $trainCmd = "python $($algo.TrainScript) --train --episodes $Episodes --save-interval $($SaveInterval) --experiment-dir `"$expDir`""
        Invoke-Expression $trainCmd

        if ($LASTEXITCODE -eq 0) {
            # 2. Run Testing (write eval results into same experiment dir)
            Write-Host "Starting testing for $($TestEpisodes) episodes..." -ForegroundColor Yellow
            $testCmd = "python $($algo.TrainScript) --test --test-episodes $TestEpisodes --save-interval $($SaveInterval) --experiment-dir `"$expDir`""
            Invoke-Expression $testCmd
        } else {
            Write-Host "Training failed for $($algo.Name)" -ForegroundColor Red
        }

        Pop-Location
    } else {
        Write-Host "Directory not found: $algoDir" -ForegroundColor Red
    }
}

Write-Host "`nAll competitive experiments completed." -ForegroundColor Cyan
