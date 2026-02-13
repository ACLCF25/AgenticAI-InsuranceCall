Set-Location 'C:\Users\AaronCarlCladoLibago\Desktop\monolith\frontend'
$out = npx tsc --noEmit --skipLibCheck 2>&1
$out | Where-Object { $_ -notlike '*\.next*' }
