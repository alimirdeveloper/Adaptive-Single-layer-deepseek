# Save as visualize_quantization.py
import matplotlib.pyplot as plt
import numpy as np

# Your results
bits = [32, 16, 8]
mape = [12.55, 12.23, 12.32]
memory_saving = [0, 50, 75]
comm_saving = [1, 2, 4]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: MAPE by bit precision
axes[0].bar(bits, mape, color=['blue', 'green', 'red'], edgecolor='black')
axes[0].set_xlabel('Bit Precision', fontsize=12)
axes[0].set_ylabel('MAPE (%)', fontsize=12)
axes[0].set_title('Quantization Impact on Accuracy', fontsize=12)
axes[0].axhline(y=12.55, color='blue', linestyle='--', alpha=0.5, label='32-bit baseline')
axes[0].legend()
for i, v in enumerate(mape):
    axes[0].text(bits[i], v + 0.2, f'{v:.2f}%', ha='center', fontsize=10)

# Plot 2: Memory savings
axes[1].bar(bits, memory_saving, color=['blue', 'green', 'red'], edgecolor='black')
axes[1].set_xlabel('Bit Precision', fontsize=12)
axes[1].set_ylabel('Memory Saving (%)', fontsize=12)
axes[1].set_title('Memory Reduction', fontsize=12)
for i, v in enumerate(memory_saving):
    axes[1].text(bits[i], v + 2, f'{v}%', ha='center', fontsize=10)

# Plot 3: Communication savings
axes[2].bar(bits, comm_saving, color=['blue', 'green', 'red'], edgecolor='black')
axes[2].set_xlabel('Bit Precision', fontsize=12)
axes[2].set_ylabel('Communication Reduction (x)', fontsize=12)
axes[2].set_title('Communication Efficiency', fontsize=12)
for i, v in enumerate(comm_saving):
    axes[2].text(bits[i], v + 0.2, f'{v}x', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('quantization_results.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("""
✅ 8-bit quantization achieves:
   • 75% memory reduction
   • 4x communication reduction  
   • Same or better accuracy than 32-bit
   
🎯 Recommendation: Use 8-bit fixed-point for resource-constrained devices!
""")