"""
tools.py - Laptop & OS Automation Tools for Kira
Allows Kira to execute terminal commands, inspect & edit files, monitor the system,
and launch apps/URLs on your laptop.
"""

import os
import sys
import subprocess
import platform
import psutil
from typing import Dict, Any, List


class LaptopTools:
    """OS automation and tool execution engine for Kira"""

    @staticmethod
    def execute_shell(command: str) -> str:
        """Run any shell or terminal command on the laptop and return output"""
        try:
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            stdout = res.stdout.strip()
            stderr = res.stderr.strip()
            if res.returncode == 0:
                return stdout if stdout else "Command completed successfully (no output)."
            else:
                return f"Error (code {res.returncode}):\n{stderr}\n{stdout}"
        except subprocess.TimeoutExpired:
            return "Command timed out after 30 seconds."
        except Exception as e:
            return f"Failed to execute command: {str(e)}"

    @staticmethod
    def read_file(filepath: str, max_lines: int = 200) -> str:
        """Read the contents of a local file on the laptop"""
        if not os.path.exists(filepath):
            return f"File not found: {filepath}"
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                lines = [f.readline() for _ in range(max_lines)]
            return "".join(lines)
        except Exception as e:
            return f"Error reading file {filepath}: {str(e)}"

    @staticmethod
    def write_file(filepath: str, content: str) -> str:
        """Create or update a local file on the laptop"""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File written successfully: {filepath}"
        except Exception as e:
            return f"Error writing file {filepath}: {str(e)}"

    @staticmethod
    def list_directory(dir_path: str = ".") -> List[str]:
        """List files and folders in a local directory"""
        try:
            if not os.path.exists(dir_path):
                return [f"Directory not found: {dir_path}"]
            items = os.listdir(dir_path)
            results = []
            for item in items:
                full_path = os.path.join(dir_path, item)
                tag = "[DIR]" if os.path.isdir(full_path) else "[FILE]"
                results.append(f"{tag} {item}")
            return results
        except Exception as e:
            return [f"Error listing directory: {str(e)}"]

    @staticmethod
    def open_url_or_app(target: str) -> str:
        """Launch a browser URL or local application"""
        try:
            if platform.system() == "Windows":
                os.startfile(target)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
            return f"Opened: {target}"
        except Exception as e:
            return f"Failed to open {target}: {str(e)}"

    @staticmethod
    def get_system_status() -> Dict[str, Any]:
        """Get CPU, Memory, Disk, and Battery usage of the laptop"""
        cpu_usage = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        battery = psutil.sensors_battery()

        return {
            "os": f"{platform.system()} {platform.release()}",
            "cpu_percent": f"{cpu_usage}%",
            "ram_used_gb": f"{ram.used / 1e9:.2f} / {ram.total / 1e9:.2f} GB ({ram.percent}%)",
            "disk_free_gb": f"{disk.free / 1e9:.2f} GB free",
            "battery_percent": f"{battery.percent}%" if battery else "No battery (Plugged in)"
        }
