targetScope = 'subscription'

@description('The name of the resource group to create.')
param rgName string = 'analyst'

@description('The location for all resources.')
param location string = 'eastus'

@description('Target environment: dev, test, or prod')
param environment string = 'dev'

@description('Name of the Key Vault used for secrets management')
param keyVaultName string = 'analyst-kv'

@description('Subscription ID for resource references (default is current subscription)')
param subscriptionId string = subscription().subscriptionId

// Create the resource group.
resource analystRG 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: rgName
  location: location
}

// Reference the existing Key Vault resource.
resource keyVault 'Microsoft.KeyVault/vaults@2021-06-01-preview' existing = {
  scope: analystRG
  name: keyVaultName
}

// =======================================================================
// Module: Virtual Network (with NSGs, subnets, and private endpoints)
// =======================================================================
module vnetModule './modules/vnet.bicep' = {
  name: 'deployVnet'
  scope: analystRG
  params: {
    location: location
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}

// ===========================================================
// Module: Log Analytics Workspace (with diagnostics enabled)
// ===========================================================
module logAnalyticsModule './modules/loga.bicep' = {
  name: 'deployLogAnalytics'
  scope: analystRG
  params: {
    workspaceName: 'analyst-loganalytics'
    location: location
    retentionInDays: 30
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}

// =======================================================================
// Module: Azure Cosmos DB Account (with VNet filtering and encryption)
// =======================================================================
module cosmosDbModule './modules/cosmos.bicep' = {
  name: 'deployCosmosDb'
  // Using the already-created resource group ensures a consistent scope.
  scope: analystRG
  params: {
    cosmosDbName: 'analyst-cosmosdb'
    location: location
    resourceGroupLocation: analystRG.location
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}

// ===================================================================================
// Module: Managed Environment for Container Apps (receives Log Analytics outputs)
// ===================================================================================
module containerAppsEnvModule './modules/aca.bicep' = {
  name: 'deployContainerAppsEnv'
  scope: analystRG
  params: {
    location: location
    infrastructureSubnetId: vnetModule.outputs.containerAppsSubnetId
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}

// ===================================================================================
// Module: Azure OpenAI Instance (with managed identity and secure secret handling)
// ===================================================================================
module azureOpenAIModule './modules/aoai.bicep' = {
  name: 'deployAzureOpenAI'
  scope: analystRG
  params: {
    location: location
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}

// =======================================================================
// Module: Azure Bing Search Resource (applying security best practices)
// =======================================================================
module bingSearch './modules/bing.bicep' = {
  name: 'deployBingSearch'
  scope: analystRG
  params: {
    location: location
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}

// =================================================================
// Module: Azure Static Web App (enforces HTTPS and CORS policies)
// =================================================================
module staticWebApp './modules/staticwapp.bicep' = {
  name: 'deployStaticWebApp'
  scope: analystRG
  params: {
    location: location
    repositoryUrl: 'https://github.com/Azure-Samples/analyst.git'
    environment: environment
    keyVaultReference: keyVault.name
  }
}

// ===================================================================================
// Module: Azure Blob Storage Account (enforces HTTPS-only and uses private endpoints)
// ===================================================================================
module storageAccount './modules/blob.bicep' = {
  name: 'deployStorageAccount'
  scope: analystRG
  params: {
    location: location
    environment: environment
    keyVaultReference: keyVault.name
    subscriptionId: subscriptionId
  }
}
