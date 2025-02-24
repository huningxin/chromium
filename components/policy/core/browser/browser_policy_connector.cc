// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/policy/core/browser/browser_policy_connector.h"

#include <stddef.h>
#include <algorithm>
#include <utility>
#include <vector>

#include "base/command_line.h"
#include "base/logging.h"
#include "base/macros.h"
#include "base/metrics/histogram_functions.h"
#include "base/metrics/histogram_macros.h"
#include "base/strings/string16.h"
#include "base/strings/utf_string_conversions.h"
#include "base/threading/thread_task_runner_handle.h"
#include "components/policy/core/common/cloud/cloud_policy_refresh_scheduler.h"
#include "components/policy/core/common/cloud/device_management_service.h"
#include "components/policy/core/common/configuration_policy_provider.h"
#include "components/policy/core/common/policy_namespace.h"
#include "components/policy/core/common/policy_pref_names.h"
#include "components/policy/core/common/policy_statistics_collector.h"
#include "components/policy/core/common/policy_switches.h"
#include "components/policy/policy_constants.h"
#include "components/prefs/pref_registry_simple.h"
#include "google_apis/gaia/gaia_auth_util.h"
#include "third_party/icu/source/i18n/unicode/regex.h"

namespace policy {

namespace {

// The URL for the device management server.
const char kDefaultDeviceManagementServerUrl[] =
    "https://m.google.com/devicemanagement/data/api";


void ReportRegexSuccessMetric(bool success) {
  UMA_HISTOGRAM_BOOLEAN("Enterprise.DomainWhitelistRegexSuccess", success);
}

// Regexes that match many of the larger public email providers as we know
// these users are not from hosted enterprise domains.
const wchar_t* const kNonManagedDomainPatterns[] = {
  L"aol\\.com",
  L"comcast\\.net",
  L"googlemail\\.com",
  L"gmail\\.com",
  L"gmx\\.de",
  L"hotmail(\\.co|\\.com|)\\.[^.]+",  // hotmail.com, hotmail.it, hotmail.co.uk
  L"live\\.com",
  L"mail\\.ru",
  L"msn\\.com",
  L"naver\\.com",
  L"orange\\.fr",
  L"outlook\\.com",
  L"qq\\.com",
  L"yahoo(\\.co|\\.com|)\\.[^.]+",  // yahoo.com, yahoo.co.uk, yahoo.com.tw
  L"yandex\\.ru",
  L"web\\.de",
  L"wp\\.pl",
  L"consumer\\.example\\.com",
};

// Returns true if |domain| matches the regex |pattern|.
bool MatchDomain(const base::string16& domain, const base::string16& pattern,
                 size_t index) {
  UErrorCode status = U_ZERO_ERROR;
  const icu::UnicodeString icu_pattern(pattern.data(), pattern.length());
  icu::RegexMatcher matcher(icu_pattern, UREGEX_CASE_INSENSITIVE, status);
  if (!U_SUCCESS(status)) {
    // http://crbug.com/365351 - if for some reason the matcher creation fails
    // just return that the pattern doesn't match the domain. This is safe
    // because the calling method (IsNonEnterpriseUser()) is just used to enable
    // an optimization for non-enterprise users - better to skip the
    // optimization than crash.
    DLOG(ERROR) << "Possible invalid domain pattern: " << pattern
                << " - Error: " << status;
    ReportRegexSuccessMetric(false);
    UMA_HISTOGRAM_ENUMERATION("Enterprise.DomainWhitelistRegexFailure",
                              index, arraysize(kNonManagedDomainPatterns));
    base::UmaHistogramSparse("Enterprise.DomainWhitelistRegexFailureStatus",
                             status);
    return false;
  }
  ReportRegexSuccessMetric(true);
  icu::UnicodeString icu_input(domain.data(), domain.length());
  matcher.reset(icu_input);
  status = U_ZERO_ERROR;
  UBool match = matcher.matches(status);
  DCHECK(U_SUCCESS(status));
  return !!match;  // !! == convert from UBool to bool.
}

}  // namespace

BrowserPolicyConnector::BrowserPolicyConnector(
    const HandlerListFactory& handler_list_factory)
    : BrowserPolicyConnectorBase(handler_list_factory) {
}

BrowserPolicyConnector::~BrowserPolicyConnector() {
}

void BrowserPolicyConnector::InitInternal(
    PrefService* local_state,
    std::unique_ptr<DeviceManagementService> device_management_service) {
  DCHECK(!is_initialized());

  device_management_service_ = std::move(device_management_service);

  policy_statistics_collector_.reset(new policy::PolicyStatisticsCollector(
      base::Bind(&GetChromePolicyDetails), GetChromeSchema(),
      GetPolicyService(), local_state, base::ThreadTaskRunnerHandle::Get()));
  policy_statistics_collector_->Initialize();

  InitPolicyProviders();
}

void BrowserPolicyConnector::Shutdown() {
  BrowserPolicyConnectorBase::Shutdown();
  device_management_service_.reset();
}

void BrowserPolicyConnector::ScheduleServiceInitialization(
    int64_t delay_milliseconds) {
  // Skip device initialization if the BrowserPolicyConnector was never
  // initialized (unit tests).
  if (device_management_service_)
    device_management_service_->ScheduleInitialization(delay_milliseconds);
}

// static
bool BrowserPolicyConnector::IsNonEnterpriseUser(const std::string& username) {
  if (username.empty() || username.find('@') == std::string::npos) {
    // An empty username means incognito user in case of ChromiumOS and
    // no logged-in user in case of Chromium (SigninService). Many tests use
    // nonsense email addresses (e.g. 'test') so treat those as non-enterprise
    // users.
    return true;
  }
  const base::string16 domain = base::UTF8ToUTF16(
      gaia::ExtractDomainName(gaia::CanonicalizeEmail(username)));
  for (size_t i = 0; i < arraysize(kNonManagedDomainPatterns); i++) {
    base::string16 pattern = base::WideToUTF16(kNonManagedDomainPatterns[i]);
    if (MatchDomain(domain, pattern, i))
      return true;
  }
  return false;
}

// static
std::string BrowserPolicyConnector::GetDeviceManagementUrl() {
  base::CommandLine* command_line = base::CommandLine::ForCurrentProcess();
  if (command_line->HasSwitch(switches::kDeviceManagementUrl))
    return command_line->GetSwitchValueASCII(switches::kDeviceManagementUrl);
  else
    return kDefaultDeviceManagementServerUrl;
}

// static
void BrowserPolicyConnector::RegisterPrefs(PrefRegistrySimple* registry) {
  registry->RegisterIntegerPref(
      policy_prefs::kUserPolicyRefreshRate,
      CloudPolicyRefreshScheduler::kDefaultRefreshDelayMs);
}


}  // namespace policy
