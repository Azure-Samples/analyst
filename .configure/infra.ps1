# Log in to Azure
Write-Host "Logging in to Azure..."

# Read the .env file. You might want to add this.
$envFile = Get-Content "src\.env"

# Set your subscription.
$subscriptionLine = $envFile | Where-Object { $_ -match "^SUBSCRIPTION_ID=" }
$subscriptionId = $subscriptionLine -replace '^SUBSCRIPTION_ID="(.+)"$', '$1'

az account set --subscription "$subscriptionId"

# Define deployment parameters.
$rgName = "analyst"
$location = "eastus2"

# Deploy the main Bicep file.
Write-Host "Deploying the Bicep template..."

try {
    az deployment sub create --location $location --template-file .\infra\main.bicep --parameters rgName=$rgName location=$location
    Write-Host "Deployment initiated."
} catch {
    Write-Error "Deployment failed: $_"
}
