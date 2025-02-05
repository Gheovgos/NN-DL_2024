import subprocess

path = 'V2.0/'
script = 'backpropago_base.py'
n = 10

def run_script():
    try:
        result = subprocess.run(['python3', path+script], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Errore nell'esecuzione dello script: {e}")
        print(f"Stderr: {e.stderr}")
        return ""

output_file = f'results_{script}.txt'

with open(output_file, 'w') as f:
    for i in range(n):
        print(f"Esecuzione {i+1}...")
        result = run_script()
        f.write(f"Esecuzione {i+1}:\n")
        f.write(result)
        f.write("\n" + "="*40 + "\n")
