"""
Heart Chat 精细日志系统
- 彩色控制台输出（支持Windows CMD ANSI）
- 多级别日志：DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
- 模块分类标签
- 时间戳精确到毫秒
- 同时写入日志文件
- 实时刷新输出
"""
import os
import sys
import time
from datetime import datetime
from typing import Optional, TextIO
from enum import Enum


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    SUCCESS = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5


# ANSI颜色代码
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # 前景色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"
    
    # 亮色前景
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    
    # 背景色
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"


# Windows CMD启用ANSI支持
def _enable_ansi_on_windows():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


_enable_ansi_on_windows()


class Logger:
    _instance: Optional['Logger'] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_dir: str = None, log_level = LogLevel.DEBUG, enable_color: bool = None):
        if self._initialized:
            return
        self._initialized = True
        
        # 兼容int类型
        if isinstance(log_level, int):
            level_map = {0: LogLevel.DEBUG, 1: LogLevel.INFO, 2: LogLevel.SUCCESS, 3: LogLevel.WARNING, 4: LogLevel.ERROR, 5: LogLevel.CRITICAL}
            log_level = level_map.get(log_level, LogLevel.DEBUG)
        self.log_level = log_level
        
        # 自动检测是否启用颜色：
        # - 显式参数优先
        # - 环境变量 NO_COLOR=1 禁用
        # - 环境变量 FORCE_COLOR=1 强制启用
        # - 否则检测是否是真实终端(isatty)
        if enable_color is None:
            if os.environ.get("NO_COLOR"):
                enable_color = False
            elif os.environ.get("FORCE_COLOR"):
                enable_color = True
            else:
                enable_color = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        self.enable_color = enable_color
        self.log_dir = log_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 日志文件：按日期命名
        self._log_file: Optional[TextIO] = None
        self._open_log_file()
        
        # 模块颜色映射
        self._module_colors = {
            "启动器": Colors.BRIGHT_CYAN,
            "服务器": Colors.BRIGHT_BLUE,
            "接口": Colors.CYAN,
            "大模型": Colors.BRIGHT_MAGENTA,
            "小模型": Colors.MAGENTA,
            "编排器": Colors.BRIGHT_GREEN,
            "解析器": Colors.GREEN,
            "身体引擎": Colors.YELLOW,
            "情绪": Colors.BRIGHT_YELLOW,
            "高潮": Colors.BRIGHT_RED,
            "感官": Colors.BRIGHT_BLUE,
            "后处理": Colors.GREEN,
            "角色": Colors.CYAN,
            "会话": Colors.BLUE,
            "世界": Colors.BRIGHT_YELLOW,
            "事件": Colors.YELLOW,
            "服装": Colors.GRAY,
            "记忆": Colors.DIM + Colors.WHITE,
            "存档": Colors.GRAY,
            "网络": Colors.BRIGHT_CYAN,
            "系统": Colors.WHITE,
        }
        
        # 级别样式
        self._level_styles = {
            LogLevel.DEBUG:    {"tag": "DEBUG", "color": Colors.GRAY, "bold": False},
            LogLevel.INFO:     {"tag": "INFO ", "color": Colors.WHITE, "bold": False},
            LogLevel.SUCCESS:  {"tag": "OK   ", "color": Colors.BRIGHT_GREEN, "bold": True},
            LogLevel.WARNING:  {"tag": "WARN ", "color": Colors.BRIGHT_YELLOW, "bold": True},
            LogLevel.ERROR:    {"tag": "ERROR", "color": Colors.BRIGHT_RED, "bold": True},
            LogLevel.CRITICAL: {"tag": "FATAL", "color": Colors.BG_RED + Colors.WHITE, "bold": True},
        }
        
        self.start_time = time.time()
    
    def _open_log_file(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(self.log_dir, f"heartchat_{date_str}.log")
        self._log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        # 写入启动分隔符
        self._write_file_raw("=" * 80)
        self._write_file_raw(f"Heart Chat 日志启动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        self._write_file_raw("=" * 80)
    
    def _print_console(self, text: str):
        """Print without letting a legacy Windows console encoding break app logic."""
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            print(text, flush=True)
        except UnicodeEncodeError:
            print(text.encode(encoding, errors="replace").decode(encoding), flush=True)

    def _write_file_raw(self, text: str):
        if self._log_file and not self._log_file.closed:
            try:
                self._log_file.write(text + "\n")
                self._log_file.flush()
            except UnicodeEncodeError:
                # 孤立代理字符等无法编码的内容：替换后重写，绝不静默丢日志
                safe = text.encode("utf-8", errors="replace").decode("utf-8")
                try:
                    self._log_file.write(safe + "\n")
                    self._log_file.flush()
                except Exception:
                    pass
            except Exception:
                pass
    
    def _strip_ansi(self, text: str) -> str:
        import re
        return re.sub(r'\033\[[0-9;]*m', '', text)
    
    def _get_module_color(self, module: str) -> str:
        if module in self._module_colors:
            return self._module_colors[module]
        # 根据模块名hash分配颜色
        colors = [Colors.CYAN, Colors.GREEN, Colors.YELLOW, Colors.BLUE, 
                  Colors.MAGENTA, Colors.BRIGHT_CYAN, Colors.BRIGHT_GREEN]
        return colors[hash(module) % len(colors)]
    
    def log(self, level: LogLevel, module: str, message: str, exc_info: bool = False):
        if level.value < self.log_level.value:
            return
        
        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S.%f")[:-3]
        uptime = time.time() - self.start_time
        uptime_str = f"{uptime:7.2f}s"
        
        style = self._level_styles[level]
        module_color = self._get_module_color(module)
        
        if self.enable_color:
            # 彩色控制台输出
            parts = []
            parts.append(Colors.GRAY + f"[{timestamp}]" + Colors.RESET)
            parts.append(Colors.DIM + f"[{uptime_str}]" + Colors.RESET)
            parts.append(style["color"] + (Colors.BOLD if style["bold"] else "") + f"[{style['tag']}]" + Colors.RESET)
            parts.append(module_color + f"[{module}]" + Colors.RESET)
            parts.append(" " + message)
            console_line = "".join(parts)
            
            if exc_info:
                import traceback
                tb_lines = []
                tb = traceback.format_exc()
                for line in tb.strip().split("\n"):
                    tb_lines.append(Colors.GRAY + "  ↳ " + line + Colors.RESET)
                tb_console = "\n".join(tb_lines)
            else:
                tb_console = None
        else:
            # 纯文本输出（无颜色，用于管道/子进程）
            parts = []
            parts.append(f"[{timestamp}]")
            parts.append(f"[{uptime_str}]")
            parts.append(f"[{style['tag'].strip()}]")
            parts.append(f"[{module}]")
            parts.append(" " + self._strip_ansi(message))
            console_line = "".join(parts)
            tb_console = None
            if exc_info:
                import traceback
                tb = traceback.format_exc()
                tb_console = "\n".join("  ↳ " + line for line in tb.strip().split("\n"))
        
        # 文件输出（始终无颜色）
        file_parts = []
        file_parts.append(f"[{timestamp}]")
        file_parts.append(f"[{uptime_str}]")
        file_parts.append(f"[{style['tag'].strip()}]")
        file_parts.append(f"[{module}]")
        file_parts.append(" " + self._strip_ansi(message))
        file_line = "".join(file_parts)
        
        # 输出到控制台
        self._print_console(console_line)
        if tb_console:
            self._print_console(tb_console)
        
        # 输出到文件
        self._write_file_raw(file_line)
        if exc_info:
            import traceback
            tb = traceback.format_exc()
            for line in tb.strip().split("\n"):
                self._write_file_raw("  ↳ " + line)
    
    # 便捷方法
    def debug(self, module: str, message: str):
        self.log(LogLevel.DEBUG, module, message)

    def trace(self, module: str, message: str):
        """仅写入日志文件，不输出控制台；用于完整Prompt、回复等大段排查内容。"""
        if self._log_file and not self._log_file.closed:
            now = datetime.now()
            timestamp = now.strftime("%H:%M:%S.%f")[:-3]
            uptime = time.time() - self.start_time
            file_line = (
                f"[{timestamp}][{uptime:7.2f}s][TRACE][{module}] "
                f"{self._strip_ansi(message)}"
            )
            self._write_file_raw(file_line)
    
    def info(self, module: str, message: str):
        self.log(LogLevel.INFO, module, message)
    
    def success(self, module: str, message: str):
        self.log(LogLevel.SUCCESS, module, message)
    
    def warning(self, module: str, message: str):
        self.log(LogLevel.WARNING, module, message)
    
    def error(self, module: str, message: str, exc_info: bool = False):
        self.log(LogLevel.ERROR, module, message, exc_info=exc_info)
    
    def critical(self, module: str, message: str, exc_info: bool = False):
        self.log(LogLevel.CRITICAL, module, message, exc_info=exc_info)
    
    def separator(self, char: str = "-", length: int = 70):
        """打印分隔线"""
        line = char * length
        if self.enable_color:
            print(Colors.GRAY + line + Colors.RESET, flush=True)
        else:
            print(line, flush=True)
        self._write_file_raw(self._strip_ansi(line))
    
    def title(self, title_text: str):
        """打印标题"""
        self.separator("=")
        if self.enable_color:
            print(f"  {Colors.BOLD}{Colors.BRIGHT_CYAN}{title_text}{Colors.RESET}", flush=True)
        else:
            print(f"  {title_text}", flush=True)
        self.separator("=")
        self._write_file_raw(f"  {title_text}")
    
    def step(self, step_num: int, total: int, message: str):
        """步骤进度"""
        self.info("启动器", f"[{step_num}/{total}] {message}")
    
    def close(self):
        if self._log_file and not self._log_file.closed:
            self._write_file_raw("-" * 80)
            self._log_file.close()


# 全局单例
_logger_instance: Optional[Logger] = None


def get_logger() -> Logger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger()
    return _logger_instance


def setup_logger(log_dir: str = None, log_level = LogLevel.DEBUG, enable_color: bool = None) -> Logger:
    global _logger_instance
    # 兼容int类型
    if isinstance(log_level, int):
        level_map = {0: LogLevel.DEBUG, 1: LogLevel.INFO, 2: LogLevel.SUCCESS, 3: LogLevel.WARNING, 4: LogLevel.ERROR, 5: LogLevel.CRITICAL}
        log_level = level_map.get(log_level, LogLevel.DEBUG)
    # 重置单例，允许重新初始化
    Logger._instance = None
    _logger_instance = Logger(log_dir=log_dir, log_level=log_level, enable_color=enable_color)
    return _logger_instance


# 模块级便捷函数
def debug(module: str, message: str):
    get_logger().debug(module, message)

def trace(module: str, message: str):
    """文件专用日志：完整记录到日志文件，不打扰控制台。"""
    get_logger().trace(module, message)

def info(module: str, message: str):
    get_logger().info(module, message)

def success(module: str, message: str):
    get_logger().success(module, message)

def warning(module: str, message: str):
    get_logger().warning(module, message)

def error(module: str, message: str, exc_info: bool = False):
    get_logger().error(module, message, exc_info=exc_info)

def critical(module: str, message: str, exc_info: bool = False):
    get_logger().critical(module, message, exc_info=exc_info)

def separator(char: str = "-", length: int = 70):
    get_logger().separator(char, length)

def title(title_text: str):
    get_logger().title(title_text)

def step(step_num: int, total: int, message: str):
    get_logger().step(step_num, total, message)
