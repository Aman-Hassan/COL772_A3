@echo off

IF "%1"=="test" (
    IF NOT "%~4"=="" (
        python A3_v1.py "test" "%2" "%3" "%4"
    ) ELSE (
        echo Error: Illegal number of parameters. Please use the format: run_model.bat test ^<path_to_data^> ^<path_to_model^> ^<path_to_result^>
        exit /b 2
    )
) ELSE IF "%1"=="train" (
    IF NOT "%~3"=="" (
        python A3_v1.py "train" "%2" "%3"
    ) ELSE (
        echo Error: Illegal number of parameters. Please use the format: run_model.bat train ^<path_to_data^> ^<path_to_save^>
        exit /b 2
    )
) ELSE (
    echo Error: Invalid argument. Please use either 'train' or 'test' as the first argument.
    exit /b 2
)