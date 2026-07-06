import os
import sys
import platform
import urllib.request
import zipfile
import tarfile

def setup_ffmpeg():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bin_dir = os.path.join(base_dir, "scripts", "bin")
    os.makedirs(bin_dir, exist_ok=True)
    
    system = platform.system().lower()
    
    if system == "windows":
        ffmpeg_exe = os.path.join(bin_dir, "ffmpeg.exe")
        ffprobe_exe = os.path.join(bin_dir, "ffprobe.exe")
        
        if os.path.exists(ffmpeg_exe) and os.path.exists(ffprobe_exe):
            print("FFmpeg and FFprobe are already installed in scripts/bin.")
            return

        print("Downloading FFmpeg for Windows...")
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        zip_path = os.path.join(bin_dir, "ffmpeg.zip")
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith('ffmpeg.exe'):
                        file_info.filename = 'ffmpeg.exe'
                        zip_ref.extract(file_info, bin_dir)
                    elif file_info.filename.endswith('ffprobe.exe'):
                        file_info.filename = 'ffprobe.exe'
                        zip_ref.extract(file_info, bin_dir)
            os.remove(zip_path)
            print("Successfully installed FFmpeg for Windows.")
        except Exception as e:
            print(f"Error downloading FFmpeg: {e}")
            
    elif system == "linux":
        ffmpeg_exe = os.path.join(bin_dir, "ffmpeg")
        ffprobe_exe = os.path.join(bin_dir, "ffprobe")
        
        if os.path.exists(ffmpeg_exe) and os.path.exists(ffprobe_exe):
            print("FFmpeg and FFprobe are already installed in scripts/bin.")
            return
            
        print("Downloading FFmpeg for Linux...")
        url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        tar_path = os.path.join(bin_dir, "ffmpeg.tar.xz")
        
        try:
            urllib.request.urlretrieve(url, tar_path)
            with tarfile.open(tar_path, "r:xz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith('/ffmpeg'):
                        member.name = 'ffmpeg'
                        tar.extract(member, bin_dir)
                    elif member.name.endswith('/ffprobe'):
                        member.name = 'ffprobe'
                        tar.extract(member, bin_dir)
            
            os.remove(tar_path)
            os.chmod(ffmpeg_exe, 0o755)
            os.chmod(ffprobe_exe, 0o755)
            print("Successfully installed FFmpeg for Linux.")
        except Exception as e:
            print(f"Error downloading FFmpeg: {e}")
            
    else:
        print(f"Automatic FFmpeg download is not configured for OS: {system}.")
        print("Please install FFmpeg manually using 'brew install ffmpeg' or your system package manager.")

if __name__ == "__main__":
    setup_ffmpeg()
