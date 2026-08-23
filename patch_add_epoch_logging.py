path = "src/train_ablation.py"
with open(path, "r") as f:
    content = f.read()

old = '''        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)'''

new = '''        print(f"    [{ablation}] run {run_idx} epoch {epoch+1}: "
              f"train_loss={train_loss:.4f} test_mf1={test_mf1*100:.2f}% "
              f"(best={best_mf1*100:.2f}%, patience={patience_counter}/{PATIENCE})",
              flush=True)

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)'''

content = content.replace(old, new)
with open(path, "w") as f:
    f.write(content)
print("Patched.")
