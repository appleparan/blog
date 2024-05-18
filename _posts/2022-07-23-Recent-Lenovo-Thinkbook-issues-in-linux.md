---
layout: post
title: Recent Lenovo Thinkbook (2022) issues in Linux
author: jongsukim
date: 2022-07-23 12:00:00 +0900
categories: [Programming]
tags:
  - AMD Ryzen CPU
  - Linux
  - Keyboard
  - Wireless
  - Kernel
  - 6000 series
  - Rembrandt
math: false
---

최근 [Lenovo Thinkbook 16 G4+ ARA](https://psref.lenovo.com/Product/ThinkBook/ThinkBook_16_G4plus_ARA)를 샀고
Arch Linux기반의 EndeavourOS를 설치하였다. 이 과정에서 삽질한 기록을 남긴다.

1. 준비물
    * 별도의 마우스
    * 별도의 키보드
    * 유선랜 연결
    * 아래의 문제들 때문에 위 준비물 없이는 리눅스 설치하기가 힘듦. 준비물을 갖추고 정상적으로 리눅스를 설치를 완료했다는 가정에서 시작

2. Wireless driver 부재
    * Thinkbook 16 G4+ ARA는 `10ec:b852` chip을 쓰고 [이 ask ubuntu post](https://askubuntu.com/questions/1412450/network-driver-for-realtek-10ecb852)와 같이 같이 별도의 드라이버를 설치해야함. 다만, EndeavourOS는 Kernel 5.18.x를 쓰고 있기 때문에 dev branch를 clone해서 컴파일 할 필요가 있음.

    ```
    #Turn off your Security Boot in BIOS

    git clone https://github.com/HRex39/rtl8852be.git -b dev
    cd rtl8852be
    make -j8
    sudo make install
    sudo modprobe 8852be
    ```

3. Keyboard 무한 반복 입력 문제
    1. 증상
        * 키보드를 누르면 바로 입력이 되지 않음
        * 두 번 누르게 되면 무한 반복으로 입력도미
        * 인터럽트가 걸려서 키보드 및 마우스가 작동이 안됨. 이 때 external 마우스와 키보드를 입력해주면 된다.
        * BIOS, GRUB, 윈도우에서 아무 문제 없지만 라이브USB 포함 부팅만 하면 문제가 생김 (Ubuntu, EndeavourOS 모두)
    2. 원인: [https://bbs.archlinux.org/viewtopic.php?id=277260](https://bbs.archlinux.org/viewtopic.php?id=277260)
        * "They made the keyboard IRQ active-low instead of the conventional active-high found in almost all other computers."
        * Lenovo뿐만 아니라, ASUS, Xiaomi 노트북에서도 AMD Ryzen Zen 3+ (Rembrandt, 6000 series) CPU를 쓰면 동일한 증상이 나타나는 것으로 보임
    3. 해결책
        * 2022/7/23 현재 BIOS 업데이트를 해도 동일한 증상 발생
            * [Lenovo Support page](https://pcsupport.lenovo.com/us/en/products/laptops-and-netbooks/thinkbook-series/thinkbook-16-g4-ara/21d1/downloads/driver-list/component?name=BIOS%2FUEFI)
            * [How to update Lenovo BIOS in Linux](https://lucraymond.net/2021/08/16/update-a-lenovo-laptop-firmware-when-you-run-linux/)
                * Hiren's BOOTCD를 사용하여 윈도우로 부팅하여 적용
                * schtask.exe없다고 오류가 뜨지만 무시
                * Live USB만들 때 [Ventoy](https://www.ventoy.net/en/index.html) 추천
        * Kernel patch 적용
            * [위 archlinux bbs link](https://bbs.archlinux.org/viewtopic.php?id=277260)에서 누군가가 [커널 패치(v5)](https://patchwork.kernel.org/project/linux-acpi/patch/20220618133712.8788-1-gch981213@gmail.com/)를 올림
            * [커널 패치(v6)](https://patchwork.kernel.org/project/linux-acpi/patch/20220712020058.90374-1-gch981213@gmail.com/)도 있지만, 빌드하고 v6의 존재를 알게되어 테스트하지 못함
            * [Kernel 5.20](https://lore.kernel.org/all/CAJZ5v0isLQVX3EqsokFthY5ka=V4Vse9T52s3EGSv41FKM1iGw@mail.gmail.com/)에서 적용예정 (9월 릴리즈 예상)
            * 한시적으로 [Patch를 적용하여 custom Kernel을 빌드](https://www.kernel.org/doc/html/v4.10/process/applying-patches.html#what-is-a-patch)해서 사용할 수 밖에 없다.
            * Build kernel
                * [https://wiki.archlinux.org/title/Kernel/Arch_Build_System](https://wiki.archlinux.org/title/Kernel/Arch_Build_System)
                * [Kernel Build시 GPG Key 이슈](https://bbs.archlinux.org/viewtopic.php?id=255968)
                    ```
                    curl -s https://keybase.io/heftig/pgp_keys.asc/?fingerprint\=a2ff3a36aaa56654109064ab19802f8b0d70fc30 | gpg --with-colons --import-options import-show --import
                    ```
                * [Patch 적용하기](https://wiki.archlinux.org/title/Patching_packages)
                * [Patch(v5)](https://patchwork.kernel.org/project/linux-acpi/patch/20220618133712.8788-1-gch981213@gmail.com/)
                    ```
                    diff --git a/drivers/acpi/resource.c b/drivers/acpi/resource.c
                    index c2d494784425..3f6a290a1060 100644
                    --- a/drivers/acpi/resource.c
                    +++ b/drivers/acpi/resource.c
                    @@ -399,6 +399,17 @@ static const struct dmi_system_id medion_laptop[] = {
                        { }
                    };

                    +static const struct dmi_system_id irq1_edge_low_shared[] = {
                    +	{
                    +		.ident = "Lenovo ThinkBook 14 G4+ ARA",
                    +		.matches = {
                    +			DMI_MATCH(DMI_SYS_VENDOR, "LENOVO"),
                    +			DMI_MATCH(DMI_BOARD_NAME, "LNVNB161216"),
                    +		},
                    +	},
                    +	{ }
                    +};
                    +
                    struct irq_override_cmp {
                        const struct dmi_system_id *system;
                        unsigned char irq;
                    @@ -409,6 +420,7 @@ struct irq_override_cmp {

                    static const struct irq_override_cmp skip_override_table[] = {
                        { medion_laptop, 1, ACPI_LEVEL_SENSITIVE, ACPI_ACTIVE_LOW, 0 },
                    +	{ irq1_edge_low_shared, 1, ACPI_EDGE_SENSITIVE, ACPI_ACTIVE_LOW, 1 },
                    };

                    static bool acpi_dev_irq_override(u32 gsi, u8 triggering, u8 polarity,

                    ```
                * [Update GRUB](https://linuxhint.com/update_grub_arch_linux-2/)
        * DSDT patch 적용
            * Xiaomi의 Redmibook에는 누군가가 DSDT 패치를 만듦 [링크1](https://zhuanlan.zhihu.com/p/530643928) [링크2](https://github.com/vrolife/modern_laptop)

4. 결론
    * Wireless driver도 설치했고, 커널 빌드해서 부팅하니까 키보드도 정상 작동
    * 위 삽질을 하고 싶지 않으면 Windows를 쓰거나 AMD Ryzen Zen 3+ (Rembrandt, 6000 series) CPU는 기다렸다가 사는 것을 추천

5. References
* [https://askubuntu.com/questions/1412450/network-driver-for-realtek-10ecb852](https://askubuntu.com/questions/1412450/network-driver-for-realtek-10ecb852)
* [https://bbs.archlinux.org/viewtopic.php?id=277260](https://bbs.archlinux.org/viewtopic.php?id=277260)
* [https://wiki.archlinux.org/title/Laptop/Lenovo#ThinkBook_series](https://wiki.archlinux.org/title/Laptop/Lenovo#ThinkBook_series)



