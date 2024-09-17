"""
Example running msmapper on Windows
Install msmapper using installer at
https://alfred.diamond.ac.uk/MSMapper/master/downloads/builds-snapshot/

1. Download "MSMapper-1.7.0.v20240513-1606-win32.x86_64-inst.exe" or equivalent
2. Run the installer, choose personal installation (only for me, not anyone on computer)
3. Add the installation folder to the PATH environment varible (Windows 10):
    a. Find MSmapper application, copy path:
        C:\\Users\\%username%\\AppData\\Local\\Programs\\MSMapper\\1.7
    b. Open Control Panel->System Properties by searching in the Windows search bar for "environment"
    and selecting "Edit the system environment variables", then click "Environment variables..." button.
    c. Under "System variables", scroll to find "path". Select "path" and click "Edit.." button
    d. In the Edit environment variable screen, click "New" and add the MsMapper application path
    e. Open a new powershell or cmd prompt, type "msmapperrc.exe" and
    ensure this runs (will give an error about a missing bean file).
"""

import os
from i16_msmapper import mapper_runner

infile = os.path.abspath('../examples/data/1041304.nxs')
outfile = infile.replace('.nxs', '_remap.nxs')
print(infile)
print(outfile)

cmd = r"C:\Users\grp66007\AppData\Local\Programs\MSMapper\1.7\msmapperc.exe -bean %s"
mapper_runner.SHELL_CMD = "msmapperrc.exe -bean %s"
mapper_runner.SHELL_CMD = cmd

mapper_runner.run_msmapper(
    input_files=[infile],
    output_file=outfile
)

print('Finished!')
