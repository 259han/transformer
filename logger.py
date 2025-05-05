import logging
import os
import sys
import time
import glob
import shutil
from pathlib import Path
# 导入统一配置
from config import LOG_CONFIG, DATA_CONFIG

# 日志级别映射
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# 获取日志目录
LOG_DIR = DATA_CONFIG['log_dir']
os.makedirs(LOG_DIR, exist_ok=True)

# 格式化日期时间，用于日志文件名
current_time = time.strftime("%Y%m%d-%H%M%S")

def cleanup_logs():
    """
    清理旧的日志文件，保留最近的n个文件
    """
    max_files = LOG_CONFIG['max_files']
    
    # 获取所有日志文件
    log_files = sorted(glob.glob(f"{LOG_DIR}/translator_*.log"), 
                      key=os.path.getmtime, 
                      reverse=True)  # 按修改时间降序排序
    
    # 如果文件数超过限制，删除旧文件
    if len(log_files) > max_files:
        for old_file in log_files[max_files:]:
            try:
                os.remove(old_file)
                print(f"清理旧日志文件: {old_file}")
            except Exception as e:
                print(f"无法删除日志文件 {old_file}: {str(e)}")

class TransformerLogger:
    def __init__(self, name="translator", level=None, console=None, log_file=None):
        # 使用配置或默认值
        self.level = level or LOG_CONFIG['level']
        self.console = console if console is not None else LOG_CONFIG['console']
        self.log_file = log_file if log_file is not None else LOG_CONFIG['log_file']
        
        # 清理旧的日志文件
        cleanup_logs()
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOG_LEVELS.get(self.level.lower(), logging.INFO))
        
        # 避免重复添加处理程序
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # 创建文件处理程序
        if self.log_file:
            log_file = Path(LOG_DIR) / f"{name}_{current_time}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
        # 如果需要，添加控制台处理程序
        if self.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)
    
    def exception(self, message):
        self.logger.exception(message)

# 创建默认日志记录器实例
logger = TransformerLogger(name="translator")

# 用于在启动时记录系统信息的函数
def log_system_info():
    """记录系统环境信息"""
    import platform
    import torch
    
    logger.info("==================== 系统信息 ====================")
    logger.info(f"操作系统: {platform.system()} {platform.version()}")
    logger.info(f"Python版本: {platform.python_version()}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU类型: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        
        # 显示GPU内存信息
        free_memory, total_memory = torch.cuda.mem_get_info(0)
        logger.info(f"GPU内存: 可用 {free_memory/(1024**3):.2f} GB / 总共 {total_memory/(1024**3):.2f} GB")
    
    logger.info("==================================================") 