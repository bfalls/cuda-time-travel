# cuda-time-travel

## Build (Windows, Visual Studio)

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Run

```
.\build\Release\tt_demo.exe
.\build\Release\tt_tests.exe
```
