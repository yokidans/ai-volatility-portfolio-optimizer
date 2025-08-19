class Logger:
    def info(self, msg, **kwargs):
        print(f"[INFO] {msg}")
        if kwargs:
            print("\t" + "\n\t".join(f"{k}: {v}" for k, v in kwargs.items()))
    
    def warning(self, msg, **kwargs):
        print(f"[WARN] {msg}")
        if kwargs:
            print("\t" + "\n\t".join(f"{k}: {v}" for k, v in kwargs.items()))
    
    def error(self, msg, **kwargs):
        print(f"[ERROR] {msg}")
        if kwargs:
            print("\t" + "\n\t".join(f"{k}: {v}" for k, v in kwargs.items()))

logger = Logger()