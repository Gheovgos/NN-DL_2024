import subprocess

path = 'V2.0/'
script = 'backpropago_ottimizzato.py'
n = 10
n_nodes = 128

def run_script():
    try:
        result = subprocess.run(['python3', script], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Errore nell'esecuzione dello script: {e}")
        print(f"Stderr: {e.stderr}")
        return ""

output_file = f'results_{script}_{n_nodes}.txt'

with open(output_file, 'w') as f:
    print(f"Eseguendo: {script} con {n_nodes}")
    print(f"Creando: {output_file}...")
    for i in range(n):
        print(f"Esecuzione {i+1}...")
        result = run_script()
        f.write(f"Esecuzione {i+1}:\n")
        f.write(result)
        f.write("\n" + "="*40 + "\n")
