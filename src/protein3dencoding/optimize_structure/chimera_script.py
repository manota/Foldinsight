from chimera import runCommand
import chimera

input_file_path=''
output_file_path=''
print(input_file_path,output_file_path)

runCommand('open '+input_file_path)
runCommand("addh")
runCommand('addcharge')
runCommand('minimize nsteps 2000 cgsteps 100')
runCommand('write #0 '+output_file_path)

