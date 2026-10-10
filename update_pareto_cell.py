import json
import sys

# Updated numbers from verified results
ACCURACY_UPD = """ACCURACY = {'PubMedQA': {'MedGemma-AR': 71.4, 'MedGemma-LL': 71.4, 'Jev': 78.8, 'MedJev': 63.8, 'Laya': 56.6, 'MedGemma-Head': 65.6, 'MedGemma-LoRA': 72.6, 'Laya-Med': 61.6}, 'MedQA': {'MedGemma-AR': 49.2, 'MedGemma-LL': 49.2, 'Jev': 86.1, 'MedJev': 28.3, 'Laya': 21.1, 'MedGemma-Head': 51.5, 'MedGemma-LoRA': 53.3, 'Laya-Med': 30.1}, 'MedMCQA': {'MedGemma-AR': 60.2, 'MedGemma-LL': 60.2, 'Jev': 85.0, 'MedJev': 35.9, 'Laya': 27.7, 'MedGemma-Head': 65.5, 'MedGemma-LoRA': 68.7, 'Laya-Med': 41.2}, 'Text Average': {'MedGemma-AR': 60.3, 'MedGemma-LL': 60.3, 'Jev': 83.3, 'MedJev': 42.7, 'Laya': 35.1, 'MedGemma-Head': 60.8, 'MedGemma-LoRA': 64.9, 'Laya-Med': 44.3}, 'VQA-RAD': {'MedGemma-AR': 70.6, 'MedGemma-LL': 70.6, 'Jev': None, 'MedJev': None, 'Laya': None, 'MedGemma-Head': 76.1, 'MedGemma-LoRA': 69.7, 'Laya-Med': 56.9}, 'SLAKE': {'MedGemma-AR': 79.4, 'MedGemma-LL': 79.4, 'Jev': None, 'MedJev': None, 'Laya': None, 'MedGemma-Head': 86.8, 'MedGemma-LoRA': 90.7, 'Laya-Med': 87.9}, 'PathVQA': {'MedGemma-AR': 62.3, 'MedGemma-LL': 62.3, 'Jev': None, 'MedJev': None, 'Laya': None, 'MedGemma-Head': 86.3, 'MedGemma-LoRA': 90.0, 'Laya-Med': 87.8}, 'PMC-VQA': {'MedGemma-AR': 48.7, 'MedGemma-LL': 48.7, 'Jev': None, 'MedJev': None, 'Laya': None, 'MedGemma-Head': 57.1, 'MedGemma-LoRA': 57.2, 'Laya-Med': 52.8}, 'Image Average': {'MedGemma-AR': 65.3, 'MedGemma-LL': 65.3, 'Jev': None, 'MedJev': None, 'Laya': None, 'MedGemma-Head': 76.6, 'MedGemma-LoRA': 76.9, 'Laya-Med': 71.3}, 'Overall Average': {'MedGemma-AR': 63.1, 'MedGemma-LL': 63.1, 'Jev': 83.3, 'MedJev': 42.7, 'Laya': 35.1, 'MedGemma-Head': 69.8, 'MedGemma-LoRA': 71.8, 'Laya-Med': 59.8}}"""

LATENCY_UPD = """LATENCY = {'PubMedQA': {'MedGemma-AR': 1034.4, 'MedGemma-LL': 50.9, 'Jev': 586.9, 'MedJev': 90.0, 'Laya': 19.6, 'MedGemma-Head': 131.5, 'MedGemma-LoRA': 144.5, 'Laya-Med': 16.1}, 'MedQA': {'MedGemma-AR': 1064.9, 'MedGemma-LL': 51.1, 'Jev': 584.0, 'MedJev': 85.5, 'Laya': 18.6, 'MedGemma-Head': 137.4, 'MedGemma-LoRA': 153.1, 'Laya-Med': 18.3}, 'MedMCQA': {'MedGemma-AR': 1049.5, 'MedGemma-LL': 34.8, 'Jev': 586.5, 'MedJev': 74.9, 'Laya': 18.1, 'MedGemma-Head': 48.7, 'MedGemma-LoRA': 54.9, 'Laya-Med': 3.9}, 'Text Average': {'MedGemma-AR': 1049.6, 'MedGemma-LL': 45.6, 'Jev': 585.8, 'MedJev': 83.5, 'Laya': 18.8, 'MedGemma-Head': 105.9, 'MedGemma-LoRA': 117.5, 'Laya-Med': 12.8}, 'VQA-RAD': {'MedGemma-AR': 1168.1, 'MedGemma-LL': 148.1, 'Jev': None, 'MedJev': None, 'Laya': None, 'MedGemma-Head': 247.9, 'MedGemma-LoRA': 255.6, 'Laya-Med': 3.5}, 'SLAKE': {'MedGemma-AR': 1159.0, 'MedGemma-LL': 151.9, 'Jev': None, 'MedJev': None, 'Laya': None, 'MedGemma-Head': 248.2, 'MedGemma-LoRA': 254.9, 'Laya-Med': 3.6}, 'PathVQA': {'MedGemma-AR': 1162.3, 'MedGemma-LL': 153.4, 'Jev': None, 'MedJev': None, 'Laya': None, 'MedGemma-Head': 249.1, 'MedGemma-LoRA': 256.8, 'Laya-Med': 3.5}, 'PMC-VQA': {'MedGemma-AR': 1160.3, 'MedGemma-LL': 147.5, 'Jev': None, 'MedJev': None, 'Laya': None, 'MedGemma-Head': 487.8, 'MedGemma-LoRA': 503.4, 'Laya-Med': 6.0}, 'Image Average': {'MedGemma-AR': 1162.4, 'MedGemma-LL': 150.2, 'Jev': None, 'MedJev': None, 'Laya': None, 'MedGemma-Head': 308.3, 'MedGemma-LoRA': 317.7, 'Laya-Med': 4.2}, 'Overall Average': {'MedGemma-AR': 1114.1, 'MedGemma-LL': 105.4, 'Jev': 585.8, 'MedJev': 83.5, 'Laya': 18.8, 'MedGemma-Head': 221.5, 'MedGemma-LoRA': 231.9, 'Laya-Med': 7.9}}"""

# Read original pareto cell
with open('/tmp/orig_pareto.py', 'r') as f:
    content = f.read()

# Replace ACCURACY block
import re
# Find the line that starts with ACCURACY = { and ends with a line that ends with }}
# We'll do a simple replacement from 'ACCURACY = {' to the line that ends with '}}' before a newline.
# We'll assume the block ends with a line that ends with '}}' and then newline.
# We'll replace from 'ACCURACY = {' up to and including the line that ends with '}}'
# We'll use a regex that matches from ACCURACY = { to the next line that ends with }} and then newline.
# Since the original has the dict spanning many lines, we'll do:
new_content = re.sub(r'ACCURACY = \{.*?\}\n', ACCURACY_UPD + '\n', content, flags=re.DOTALL)
# Replace LATENCY block similarly
new_content = re.sub(r'LATENCY = \{.*?\}\n', LATENCY_UPD + '\n', new_content, flags=re.DOTALL)

with open('/tmp/orig_pareto_updated.py', 'w') as f:
    f.write(new_content)
print('Updated pareto cell written to /tmp/orig_pareto_updated.py')